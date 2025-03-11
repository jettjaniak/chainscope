#!/usr/bin/env python3
import argparse
import pickle
import random
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import wandb
from dacite import from_dict
from requests.exceptions import HTTPError as RequestsHTTPError
from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from wandb.apis.public.files import File as WandbFile
from wandb.apis.public.runs import Run
from wandb.errors import CommError as WandbCommError
from wandb.sdk.wandb_run import Run as WandbSdkRun

from chainscope.typing import *
from chainscope.utils import get_git_commit_hash, setup_determinism


@dataclass(kw_only=True)
class ProbeConfig:
    d_model: int
    weight_init_range: float
    weight_init_seed: int


ResidByQidByTemplate = dict[str, dict[str, Float[torch.Tensor, "model"]]]


@dataclass
class DataConfig:
    model_name: str
    layer: int
    loc: str
    cv_seed: int
    cv_n_folds: int
    cv_test_fold: int
    train_val_seed: int
    val_frac: float
    device: str
    batch_size: int

    def load_resids_by_qid_by_template(self, resids_dir: Path) -> ResidByQidByTemplate:
        # Find all dataset directories
        # Map from template to (yes_dataset, no_dataset) pairs
        template_to_dataset_ids: dict[str, tuple[str | None, str | None]] = {}
        resid_by_qid_by_template = {}

        for dataset_dir in resids_dir.iterdir():
            if not dataset_dir.is_dir() or not dataset_dir.name.startswith("wm-"):
                continue
            model_dir = dataset_dir / self.model_name
            if not model_dir.exists():
                raise ValueError(f"Model directory not found: {model_dir}")

            # Extract prop_id, comparison, and yes/no from dataset ID
            dataset_id = dataset_dir.name
            parts = dataset_id.split("_")
            assert len(parts) >= 3, f"Dataset ID {dataset_id} has less than 3 parts"

            prop_id, comparison = parts[0], parts[1]
            template = f"{prop_id}_{comparison}"
            is_yes = parts[2] == "YES"

            # Update the template's dataset pair
            if template not in template_to_dataset_ids:
                template_to_dataset_ids[template] = (None, None)

            yes_dataset_id, no_dataset_id = template_to_dataset_ids[template]
            if is_yes:
                if yes_dataset_id is not None:
                    raise ValueError(
                        f"Found duplicate YES dataset for template {template}: {yes_dataset_id} and {dataset_id}"
                    )
                template_to_dataset_ids[template] = (dataset_id, no_dataset_id)
            else:
                if no_dataset_id is not None:
                    raise ValueError(
                        f"Found duplicate NO dataset for template {template}: {no_dataset_id} and {dataset_id}"
                    )
                template_to_dataset_ids[template] = (yes_dataset_id, dataset_id)

        # Load and merge residuals for complete pairs
        for template, (
            yes_dataset_id,
            no_dataset_id,
        ) in template_to_dataset_ids.items():
            assert (
                yes_dataset_id is not None
            ), f"YES dataset ID is None for template {template}"
            assert (
                no_dataset_id is not None
            ), f"NO dataset ID is None for template {template}"

            # Load YES residuals
            yes_layer_file = (
                resids_dir / yes_dataset_id / self.model_name / f"L{self.layer:02d}.pkl"
            )
            assert yes_layer_file.exists()

            # Load NO residuals
            no_layer_file = (
                resids_dir / no_dataset_id / self.model_name / f"L{self.layer:02d}.pkl"
            )
            assert no_layer_file.exists()

            # Merge residuals
            with open(yes_layer_file, "rb") as f:
                yes_resid_by_loc_by_qid = pickle.load(f)
                yes_resid_by_qid = {
                    k: v[self.loc].to(self.device).float()
                    for k, v in yes_resid_by_loc_by_qid.items()
                }
            with open(no_layer_file, "rb") as f:
                no_resid_by_loc_by_qid = pickle.load(f)
                no_resid_by_qid = {
                    k: v[self.loc].to(self.device).float()
                    for k, v in no_resid_by_loc_by_qid.items()
                }
            n_yes_resid = len(yes_resid_by_qid)
            n_no_resid = len(no_resid_by_qid)
            assert (
                n_yes_resid == n_no_resid
            ), f"YES and NO residuals have different lengths for template {template}. YES: {n_yes_resid}, NO: {n_no_resid}"

            # Combine both sets of residuals
            resid_by_qid_by_template[template] = yes_resid_by_qid | no_resid_by_qid
            total_len = len(resid_by_qid_by_template[template])
            assert (
                total_len == n_yes_resid + n_no_resid
            ), f"Total length of residuals for template {template} is not equal to the sum of YES and NO lengths. Total: {total_len}, YES: {n_yes_resid}, NO: {n_no_resid}"

        if not resid_by_qid_by_template:
            raise ValueError(
                f"No complete YES/NO pairs found for model {self.model_name} layer {self.layer}"
            )

        return resid_by_qid_by_template


def load_resids_and_df(data_config: DataConfig, resids_dir: Path):
    resid_by_qid_by_template = data_config.load_resids_by_qid_by_template(resids_dir)
    df = pd.read_pickle(DATA_DIR / "df-wm.pkl")
    model_ids = [
        mid for mid in df.model_id.unique() if mid.endswith(data_config.model_name)
    ]
    assert len(model_ids) == 1
    model_id = model_ids[0]
    df = df[df.model_id == model_id]
    return resid_by_qid_by_template, df


@dataclass
class CollateFnOutput:
    resids: Float[torch.Tensor, "batch model"]
    labels: Float[torch.Tensor, "batch"]


@dataclass
class TrainerConfig:
    probe_config: ProbeConfig
    data_config: DataConfig
    lr: float
    beta1: float
    beta2: float
    patience: int
    max_steps: int
    device: str
    experiment_uuid: str
    val_freq: int

    def get_run_name(self) -> str:
        dc = self.data_config
        assert (
            dc.train_val_seed == self.probe_config.weight_init_seed
        ), f"Train/val seed {dc.train_val_seed} does not match weight init seed {self.probe_config.weight_init_seed}"
        train_seed = dc.train_val_seed
        args = [
            f"L{dc.layer:02d}",
            f"{dc.loc}",
            f"cvs{dc.cv_seed}",
            f"ts{train_seed}",
            self.experiment_uuid,
            f"f{dc.cv_test_fold}",
        ]
        return "_".join(args)


class LinearProbe(nn.Module):
    def __init__(self, c: ProbeConfig):
        super().__init__()
        self.c = c
        setup_determinism(c.weight_init_seed)
        self.w = nn.Parameter(torch.randn(c.d_model) * c.weight_init_range)
        self.b = nn.Parameter(torch.zeros(1))

    @property
    def device(self) -> torch.device:
        return self.w.device

    def forward(
        self,
        resids: Float[torch.Tensor, "batch model"],
    ) -> Float[torch.Tensor, "batch"]:
        return self.w @ resids.T + self.b


def collate_fn(
    batch: Sequence[tuple[Float[torch.Tensor, "d_model"], float]],
) -> CollateFnOutput:
    resids_list = []
    labels_list = []
    for resid, label in batch:
        resids_list.append(resid)
        labels_list.append(label)
    device = resids_list[0].device
    return CollateFnOutput(
        resids=torch.stack(resids_list),
        labels=torch.tensor(labels_list, device=device, dtype=torch.float32),
    )


class SequenceDataset(Dataset):
    def __init__(
        self,
        resids: list[Float[torch.Tensor, "model"]],
        labels: list[float],
    ):
        self.resids = resids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Float[torch.Tensor, "model"], float]:
        return self.resids[idx], self.labels[idx]


def make_dataset_from_templates(
    templates_: list[str],
    resid_by_qid_by_template: ResidByQidByTemplate,
    df: pd.DataFrame,
) -> SequenceDataset:
    resids_list = []
    labels_list = []
    for t in templates_:
        prop_id, comparison = t.split("_")
        template_bias = df[
            (df["prop_id"] == prop_id) & (df["comparison"] == comparison)
        ]["p_yes"].mean()
        resid_by_qid = resid_by_qid_by_template[t]
        resids_list += list(resid_by_qid.values())
        labels_list += [template_bias] * len(resid_by_qid)
    n = len(resids_list)
    shuffled_idxs = random.sample(range(n), n)
    resids_list = [resids_list[i] for i in shuffled_idxs]
    labels_list = [labels_list[i] for i in shuffled_idxs]
    return SequenceDataset(resids_list, labels_list)


def load_and_split_data(
    resid_by_qid_by_template: ResidByQidByTemplate,
    df: pd.DataFrame,
    data_config: DataConfig,
) -> tuple[
    tuple[DataLoader, DataLoader, dict[str, DataLoader]],
    tuple[list[str], list[str], list[str]],
]:
    def make_dataset(templates_: list[str]) -> SequenceDataset:
        return make_dataset_from_templates(templates_, resid_by_qid_by_template, df)

    # Group templates by prop_id
    templates = list(resid_by_qid_by_template.keys())
    prop_id_to_templates: dict[str, list[str]] = {}
    for template in templates:
        prop_id, _ = template.split("_")
        if prop_id not in prop_id_to_templates:
            prop_id_to_templates[prop_id] = []
        prop_id_to_templates[prop_id].append(template)

    # Get list of unique prop_ids and shuffle them for splitting
    prop_ids = list(prop_id_to_templates.keys())
    setup_determinism(data_config.cv_seed)
    random.shuffle(prop_ids)

    # Split prop_ids into test and train_val sets
    n_folds = data_config.cv_n_folds
    test_fold_nr = data_config.cv_test_fold
    test_size = len(prop_ids) // n_folds
    test_start = test_fold_nr * test_size
    test_end = (test_fold_nr + 1) * test_size
    test_prop_ids = prop_ids[test_start:test_end]
    train_val_prop_ids = prop_ids[:test_start] + prop_ids[test_end:]

    # Split train_val prop_ids into train and val
    setup_determinism(data_config.train_val_seed)
    random.shuffle(train_val_prop_ids)
    val_size = int(len(train_val_prop_ids) * data_config.val_frac)
    val_prop_ids = train_val_prop_ids[:val_size]
    train_prop_ids = train_val_prop_ids[val_size:]

    # Map prop_ids back to their templates
    test_templates = [t for pid in test_prop_ids for t in prop_id_to_templates[pid]]
    val_templates = [t for pid in val_prop_ids for t in prop_id_to_templates[pid]]
    train_templates = [t for pid in train_prop_ids for t in prop_id_to_templates[pid]]

    # Create datasets
    train_dataset = make_dataset(train_templates)
    val_dataset = make_dataset(val_templates)

    # Create individual test datasets and loaders for each template
    test_loaders: dict[str, DataLoader] = {}
    for template in test_templates:
        test_dataset = make_dataset([template])
        test_loaders[template] = DataLoader(
            test_dataset,
            batch_size=len(test_dataset),
            collate_fn=collate_fn,
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    # no batching for validation set
    val_loader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        collate_fn=collate_fn,
    )

    return (train_loader, val_loader, test_loaders), (
        train_templates,
        val_templates,
        test_templates,
    )


def compute_test_losses_predictions(
    model: LinearProbe,
    criterion: nn.MSELoss,
    test_loaders: dict[str, DataLoader],
) -> tuple[dict[str, float], dict[str, tuple[float, float]]]:
    model.eval()
    losses = {}
    predictions = {}
    for template, loader in test_loaders.items():
        batch = next(iter(loader))
        model_output = model(batch.resids)
        loss = criterion(model_output, batch.labels).item()
        losses[template] = loss
        mean_pred = model_output.mean().item()
        std_pred = model_output.std().item()
        predictions[template] = (mean_pred, std_pred)
        assert isinstance(mean_pred, float) and isinstance(std_pred, float)
    return losses, predictions


class ProbeTrainer:
    def __init__(
        self,
        *,
        c: TrainerConfig,
        resid_by_qid_by_template: ResidByQidByTemplate,
        df: pd.DataFrame,
        model_state_dict: dict[str, Any] | None = None,
    ):
        self.c = c
        self.device = torch.device(c.device)
        self.criterion = nn.MSELoss()

        # Initialize model
        self.model = LinearProbe(self.c.probe_config).to(self.device)
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)

        # Create data loaders
        (
            (
                self.train_loader,
                self.val_loader,
                self.test_loaders,
            ),
            (
                self.train_templates,
                self.val_templates,
                self.test_templates,
            ),
        ) = load_and_split_data(
            resid_by_qid_by_template,
            df,
            self.c.data_config,
        )

    @classmethod
    def from_run(cls, run: Run, resids_dir: Path) -> "ProbeTrainer":
        del run.config["git_commit"]
        trainer_config = from_dict(TrainerConfig, run.config)

        # Download and load model weights
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "best_model.pt"
            best_model_file: WandbFile = run.file("best_model.pt")  # type: ignore
            best_model_file.download(root=tmp_dir, replace=True)
            model_state_dict = torch.load(
                tmp_path, map_location="cpu", weights_only=True
            )

        resid_by_qid_by_template, df = load_resids_and_df(
            trainer_config.data_config, resids_dir
        )

        trainer = cls(
            c=trainer_config,
            resid_by_qid_by_template=resid_by_qid_by_template,
            df=df,
            model_state_dict=model_state_dict,
        )
        return trainer

    @classmethod
    def from_wandb(
        cls,
        resids_dir: Path,
        entity: str = "cot-probing",
        project: str = "bias-probes",
        run_id: str | None = None,
        config_filters: dict[str, Any] | None = None,
    ) -> list[tuple["ProbeTrainer", Run]]:
        api = wandb.Api()

        if run_id is not None:
            assert (
                config_filters is None
            ), "Must specify exactly one of run_id or config_filters, specified both"
            run = api.run(f"{entity}/{project}/{run_id}")
        else:
            assert (
                config_filters is not None
            ), "Must specify exactly one of run_id or config_filters, specified none"
            filters = []
            for k, v in config_filters.items():
                filters.append({f"config.{k}": v})
            runs = list(api.runs(f"{entity}/{project}", {"$and": filters}))

            if len(runs) == 0:
                raise ValueError("No runs found matching config filters")

        ret = []
        for run in runs:
            assert run.state == "finished"
            ret.append((cls.from_run(run, resids_dir), run))
        return ret

    @classmethod
    def wandb_runs(
        cls,
        config_filters: dict[str, Any],
        entity: str = "cot-probing",
        project: str = "bias-probes",
        n_runs: int | None = None,
    ) -> list[Run]:
        api = wandb.Api(timeout=120)
        filters = [{"state": "finished"}]
        for k, v in config_filters.items():
            filters.append({f"config.{k}": v})

        for i in range(10):
            try:
                runs = api.runs(f"{entity}/{project}", {"$and": filters})
                if n_runs is not None and len(runs) != n_runs:
                    raise ValueError(f"Expected {n_runs} runs, got {len(runs)}")
                break
            except (RequestsHTTPError, WandbCommError):
                print(f"Error fetching runs, retrying... ({i + 1}/10)")
                pass
        else:
            raise Exception("Failed to fetch runs")

        ret = []
        for run in runs:
            if run.state != "finished":
                raise Exception(f"Run {run.id} is not finished")
            ret.append(run)
        return ret

    def train(
        self,
        project_name: str,
    ) -> tuple[LinearProbe, WandbSdkRun]:
        # Initialize W&B
        run = wandb.init(
            entity="cot-probing",
            project=project_name,
            name=self.c.get_run_name(),
        )
        wandb.config.update(asdict(self.c))
        wandb.config.update({"git_commit": get_git_commit_hash()})

        optimizer = Adam(
            self.model.parameters(), lr=self.c.lr, betas=(self.c.beta1, self.c.beta2)
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            best_model_state = train_probe(
                model=self.model,
                optimizer=optimizer,
                criterion=self.criterion,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                max_steps=self.c.max_steps,
                val_freq=self.c.val_freq,
                patience=self.c.patience,
            )

            # Save the best model
            save_subdir = tmp_dir_path / "save_best"
            save_subdir.mkdir(exist_ok=True)
            tmp_path = save_subdir / "best_model.pt"
            torch.save(best_model_state, tmp_path)
            wandb.save(str(tmp_path), base_path=str(save_subdir))

            self.model.load_state_dict(best_model_state)
            # Log individual template losses and predictions
            test_losses, test_predictions = compute_test_losses_predictions(
                self.model, self.criterion, self.test_loaders
            )
            wandb.log(
                {
                    f"test_loss/{template}": loss
                    for template, loss in test_losses.items()
                }
            )
            wandb.log(
                {
                    f"test_prediction_mean/{template}": pred[0]
                    for template, pred in test_predictions.items()
                }
            )
            wandb.log(
                {
                    f"test_prediction_std/{template}": pred[1]
                    for template, pred in test_predictions.items()
                }
            )
            wandb.finish()

        return self.model, run


def collate_fn_output_to_loss(
    model: LinearProbe,
    criterion: nn.MSELoss,
    collate_fn_output: CollateFnOutput,
) -> Float[torch.Tensor, ""]:
    model_output = model(collate_fn_output.resids)
    assert (
        model_output.dtype == collate_fn_output.labels.dtype
    ), f"Model output dtype {model_output.dtype} does not match labels dtype {collate_fn_output.labels.dtype}"
    loss = criterion(model_output, collate_fn_output.labels)
    return loss


def compute_loss_one_batch_loader(
    model: LinearProbe,
    criterion: nn.MSELoss,
    loader: DataLoader,
) -> float:
    model.eval()
    return collate_fn_output_to_loss(model, criterion, next(iter(loader))).item()


def validate(
    model: LinearProbe,
    criterion: nn.MSELoss,
    val_loader: DataLoader,
) -> float:
    model.eval()
    val_loss = compute_loss_one_batch_loader(model, criterion, val_loader)
    model.train()
    return val_loss


def train_probe(
    model: LinearProbe,
    optimizer: Adam,
    criterion: nn.MSELoss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_steps: int,
    val_freq: int,
    patience: int,
) -> dict[str, Any]:
    model.train()
    train_loss = 0
    best_val_loss = float("inf")
    best_model_state = deepcopy(model.state_dict())
    patience_counter = 0
    n_steps = 0

    train_iter = iter(train_loader)
    pbar = tqdm(total=max_steps, desc="Training")

    while n_steps < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        optimizer.zero_grad()
        loss = collate_fn_output_to_loss(model, criterion, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_steps += 1
        pbar.update(1)

        # Run validation every val_freq steps
        if n_steps % val_freq == 0:
            avg_train_loss = train_loss / val_freq
            train_loss = 0  # Reset accumulator

            val_loss = validate(model, criterion, val_loader)
            avg_val_train_loss = val_loss * 0.75 + avg_train_loss * 0.25
            wandb.log(
                {
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "avg_val_train_loss": avg_val_train_loss,
                    "step": n_steps,
                }
            )
            if avg_val_train_loss < best_val_loss:
                best_val_loss = avg_val_train_loss
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += val_freq
                if patience_counter >= patience:
                    print(f"Early stopping at step {n_steps}")
                    break

    pbar.close()
    return best_model_state


def build_data_config(args: argparse.Namespace, cv_test_fold: int) -> DataConfig:
    return DataConfig(
        model_name=args.model_name,
        layer=args.layer,
        loc=args.loc,
        cv_seed=args.cv_seed,
        cv_n_folds=args.cv_n_folds,
        cv_test_fold=cv_test_fold,
        train_val_seed=args.train_seed,
        val_frac=args.val_frac,
        device=args.device,
        batch_size=args.batch_size,
    )


def build_trainer_config(
    args: argparse.Namespace,
    experiment_uuid: str,
    cv_test_fold: int,
) -> TrainerConfig:
    probe_config = ProbeConfig(
        d_model=args.d_model,
        weight_init_range=args.weight_init_range,
        weight_init_seed=args.train_seed,
    )

    data_config = build_data_config(args, cv_test_fold)

    return TrainerConfig(
        probe_config=probe_config,
        data_config=data_config,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        patience=args.patience,
        max_steps=args.max_steps,
        device=args.device,
        experiment_uuid=experiment_uuid,
        val_freq=args.val_freq,
    )


def train_bias_probe(
    args: argparse.Namespace,
    experiment_uuid: str,
    cv_test_fold: int,
) -> tuple[LinearProbe, WandbSdkRun]:
    trainer_config = build_trainer_config(args, experiment_uuid, cv_test_fold)
    resid_by_qid_by_template, df = load_resids_and_df(
        trainer_config.data_config, args.resids_dir
    )
    trainer = ProbeTrainer(
        c=trainer_config,
        resid_by_qid_by_template=resid_by_qid_by_template,
        df=df,
    )
    return trainer.train(project_name=args.wandb_project)
