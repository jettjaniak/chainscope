import re

def extract_costs(text):
    # Use regex to find numbers that appear just before '$'
    pattern = r'(\d+\.\d+)\s*\$'
    costs = re.findall(pattern, text)
    
    # Convert strings to floats
    costs = [float(cost) for cost in costs]
    
    return costs

def main():
    # Your log text here as a string
    log_text = """    Model    App    
Tokens
Cost
Speed
Provider
Jan 22, 10:15:50 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,029
584
0.0358
$
43.3
tps
Amazon Bedrock    
Jan 22, 10:15:48 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,195
537
0.0356
$
44.5
tps
Amazon Bedrock    
Jan 22, 10:15:48 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,024
443
0.0337
$
37.8
tps
Amazon Bedrock    
Jan 22, 10:15:48 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,063
474
0.0343
$
40.3
tps
Amazon Bedrock    
Jan 22, 10:15:46 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,113
366
0.0328
$
39.2
tps
Amazon Bedrock    
Jan 22, 10:15:46 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,424
369
0.0338
$
41.7
tps
Amazon Bedrock    
Jan 22, 10:15:45 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,019
317
0.0318
$
35.9
tps
Amazon Bedrock    
Jan 22, 10:15:45 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,109
344
0.0325
$
38.3
tps
Amazon Bedrock    
Jan 22, 10:15:45 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,071
400
0.0332
$
42.9
tps
Amazon Bedrock    
Jan 22, 10:15:45 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,134
416
0.0336
$
46.3
tps
Amazon Bedrock    
Jan 22, 10:15:45 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
8,986
527
0.0349
$
63.3
tps
Anthropic    
Jan 22, 10:15:44 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
8,979
362
0.0324
$
42.0
tps
Amazon Bedrock    
Jan 22, 10:15:44 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
8,967
512
0.0346
$
44.3
tps
Amazon Bedrock    
Jan 22, 10:15:44 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,045
413
0.0333
$
60.6
tps
Anthropic    
Jan 22, 10:15:44 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,012
441
0.0337
$
36.7
tps
Amazon Bedrock    
Jan 22, 10:15:44 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,152
340
0.0326
$
54.9
tps
Anthropic    
Jan 22, 10:15:44 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,002
490
0.0344
$
68.2
tps
Anthropic    
Jan 22, 10:15:44 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,087
313
0.032
$
40.1
tps
Amazon Bedrock    
Jan 22, 10:15:44 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,042
443
0.0338
$
62.0
tps
Anthropic    
Jan 22, 10:15:44 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,085
363
0.0327
$
43.2
tps
Amazon Bedrock
and


Copy
Timestamp    Model    App    
Tokens
Cost
Speed
Provider
Jan 22, 10:15:43 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,006
328
0.0319
$
53.8
tps
Anthropic    
Jan 22, 10:15:43 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,006
498
0.0345
$
72.4
tps
Anthropic    
Jan 22, 10:15:43 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,062
560
0.0356
$
68.9
tps
Anthropic    
Jan 22, 10:15:42 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,034
504
0.0347
$
46.6
tps
Amazon Bedrock    
Jan 22, 10:15:42 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,117
357
0.0327
$
56.5
tps
Anthropic    
Jan 22, 10:15:42 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,141
394
0.0333
$
55.1
tps
Anthropic    
Jan 22, 10:15:42 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,029
430
0.0335
$
43.2
tps
Amazon Bedrock    
Jan 22, 10:15:42 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,214
386
0.0334
$
60.2
tps
Anthropic    
Jan 22, 10:15:41 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,216
424
0.034
$
44.2
tps
Amazon Bedrock    
Jan 22, 10:15:41 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,024
413
0.0333
$
45.5
tps
Amazon Bedrock    
Jan 22, 10:15:41 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,078
343
0.0324
$
54.4
tps
Anthropic    
Jan 22, 10:15:41 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,051
529
0.0351
$
56.8
tps
Amazon Bedrock    
Jan 22, 10:15:41 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,165
333
0.0325
$
38.2
tps
Amazon Bedrock    
Jan 22, 10:15:39 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,051
324
0.032
$
43.8
tps
Amazon Bedrock    
Jan 22, 10:15:39 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,019
468
0.0341
$
58.9
tps
Anthropic    
Jan 22, 10:15:39 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,029
419
0.0334
$
53.4
tps
Amazon Bedrock    
Jan 22, 10:15:39 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,068
383
0.0329
$
55.2
tps
Anthropic    
Jan 22, 10:15:39 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
8,966
481
0.0341
$
61.5
tps
Anthropic    
Jan 22, 10:15:39 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,211
347
0.0328
$
49.9
tps
Amazon Bedrock    
Jan 22, 10:15:38 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,014
439
0.0336
$
56.3
tps
Anthropic
Jan 22, 10:15:38 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,111
376
0.033
$
60.0
tps
Anthropic    
Jan 22, 10:15:37 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
9,277
354
0.0331
$
60.7
tps
Anthropic    
Jan 22, 10:15:35 PM    
Anthropic: Claude 3.5 Sonnet

Unknown
8,911
262
0.0307
$
58.2
tps
Anthropic"""
    
    # Extract and sum costs
    costs = extract_costs(log_text)
    total_cost = sum(costs)
    
    print(f"Found {len(costs)} cost entries")
    print(f"Total cost: ${total_cost:.2f}")
    print("\nIndividual costs:")
    for cost in costs:
        print(f"${cost:.2f}")

if __name__ == "__main__":
    main()
