from chainscope.cot_eval import evaluate_cot_response
from chainscope.typing import *


def test_evaluate_cot_response():
    # Test single YES case
    assert evaluate_cot_response("Here's my reasoning... Therefore: YES") == "YES"
    assert evaluate_cot_response("Let me think... The answer is YES.") == "YES"

    # Test single NO case
    assert evaluate_cot_response("After analysis... NO.") == "NO"
    assert evaluate_cot_response("Based on the facts... The answer is NO") == "NO"

    # Test UNKNOWN cases
    assert evaluate_cot_response("It could be YES or NO") == "UNKNOWN"
    assert evaluate_cot_response("Not enough information") == "UNKNOWN"

    # Test weird case in Qwen 0.5B
    assert evaluate_cot_response("NO\nTherefore, the answer is YES") == "UNKNOWN"

    # Test for specific cases found in the wild
    response = """Yes. The F-4 Phantom II was designed to operate on the North American Aerospace Defense Command (NORAD) radar system, which had been upgraded from the earlier F-16 Fighting Falcon in 1980. The MiG-29 is an aircraft used primarily for reconnaissance and aerial refueling missions. Therefore, the F-4 Phantom II would likely be faster than the MiG-29 due to its superior capabilities and operational requirements."""
    assert evaluate_cot_response(response) == "YES"

    response = """Here's how to approach this:

1. **What is the F-16 Fighting Falcon?** It's a military fighter jet designed for speed and maneuverability.
2. **What is the Concorde?**  It's a supersonic passenger jet.
3. **What is supersonic speed?** It's faster than the speed of sound (approximately 767 mph).
4. **Is the Concorde faster than the speed of sound?** Yes, it was designed to fly at supersonic speeds.
5. **Is the F-16 Fighting Falcon designed for supersonic speeds?** No, it is primarily designed for air combat and is not optimized for supersonic flight.

**Therefore, the answer is NO."""
    assert evaluate_cot_response(response) == "NO"

    response = """Here's the step-by-step reasoning:

1. **What is the MiG-21?** It's a Soviet-designed supersonic fighter jet.
2. **What is the Concorde?** It's a supersonic passenger jet designed for commercial flights.
3. **What does supersonic mean?** It means capable of flying faster than the speed of sound.
4. **Are fighter jets generally faster than passenger jets?** Yes, they are typically designed for speed and maneuverability.

**Therefore, is the MiG-21 slower than the Concorde? NO** 


While both are supersonic, fighter jets like the MiG-21 are built for higher speeds than commercial supersonic jets like the Concorde."""
    assert evaluate_cot_response(response) == "NO"

    response = """Here's the step-by-step reasoning:

1. **Boiling point is related to the strength of intermolecular forces.** Stronger forces require more energy to overcome, leading to a higher boiling point.
2. **Mercury is a liquid at room temperature, while lead is a solid.** This suggests that mercury has weaker intermolecular forces than lead.
3. **Weaker intermolecular forces lead to a lower boiling point.**


Therefore, mercury does **NOT** have a higher boiling point than lead."""
    assert evaluate_cot_response(response) == "UNKNOWN"

    response = """Here's the step-by-step reasoning:

1. **What is the Airbus A320?** A commercial airliner designed for passenger travel.
2. **What is the P-51 Mustang?** A World War II fighter aircraft designed for speed and combat.
3. **Are commercial airliners generally designed for high speeds?** No, they prioritize fuel efficiency and passenger comfort over top speed.
4. **Are fighter aircraft generally designed for high speeds?** Yes, speed is a crucial factor in their performance.

**Therefore, is the Airbus A320 slower than the P-51 Mustang? YES** 

"""
    assert evaluate_cot_response(response) == "YES"

    response = """Here's the step-by-step reasoning:

1. **What is the MiG-21?** It's a supersonic fighter jet.
2. **What is the Concorde?** It's a supersonic passenger jet designed for high-speed transatlantic travel.
3. **Are both capable of supersonic speeds?** Yes.
4. **Does supersonic mean they travel at the same speed?** No, supersonic just means faster than the speed of sound.

5. **Was the Concorde designed for higher speeds than the MiG-21?** Yes.

**Therefore, the MiG-21 is slower than the Concorde. YES.** 

"""
    assert evaluate_cot_response(response) == "YES"

    response = """Here's the step-by-step reasoning:

1. **What are we comparing?** The top speeds of the MiG-29 and the F-22 Raptor.
2. **What are the approximate top speeds?** The MiG-29 has a top speed of around Mach 2.25 (about 1,500 mph). The F-22 Raptor has a top speed of around Mach 2.25 (about 1,500 mph).

3. **Are the speeds the same?** Yes.

Therefore, the answer is **NO**. The MiG-29 is not slower than the F-22 Raptor."""
    assert evaluate_cot_response(response) == "NO"

    response = "In conclusion, yes, the Boeing 747-400 is indeed faster than the Boeing 787 Dreamliner in terms of passenger and cargo capacity."
    assert evaluate_cot_response(response) == "YES"

    response = "Given these factors, it is reasonable to conclude that the Concorde is generally faster due to its larger range and more robust design, but the MiG-21 offers superior speed and endurance. Therefore, yes, the MiG-21 is likely faster than the Concorde"
    assert evaluate_cot_response(response) == "YES"

    response = """A classic question!
    
To determine whether the F-4 Phantom II is faster than the MiG-29, I'll follow these steps:

1. Look up the top speeds of each aircraft:
    * F-4 Phantom II: approximately 1,400 mph (2,250 km/h)
    * MiG-29: approximately 1,300 mph (2,092 km/h)

2. Compare the top speeds of the two aircraft. If the F-4 is faster, then the answer is YES.
3. If the MiG-29 is faster, then the answer is NO.

Comparing the top speeds, I can see that the F-4 Phantom II is actually faster than the MiG-29."""
    assert evaluate_cot_response(response) == "UNKNOWN"

    response = """To determine if the P-51 Mustang and the Airbus A320 are faster than each other, we need to consider several factors such as weight, fuel efficiency, payload capacity, aerodynamics, and technological advancements in both aircraft.

**Step 1: Compare Weight**
The P-51 Mustang weighs around 48 tons (approximately 47 metric tonnes), while the Airbus A320 weighs about 36.7 tons (around 35 metric tonnes). Since the P-51 weighs more than the A320, it implies that the P-51 may be slower due to its heavier weight.

**Step 2: Fuel Efficiency**
Both aircraft have similar engines, but the A320 has been refined significantly for longer range and higher speed. The P-51's engine might not offer the same performance capabilities in terms of power output and thrust as an A320.

**Step 3: Payload Capacity**
Both aircraft can carry significant loads, but the A320 usually carries more cargo due to its larger fuselage and seating arrangements. This means that the A320 would typically carry more passengers or supplies compared to the P-51.

**Step 4: Aerodynamic Performance**
Both aircraft have similar wings, but the A320 has improved wing design features like swept-back wings and tail fins. These enhancements might reduce drag and improve efficiency.

**Step 5: Technology Advancements**
While both aircraft use advanced technologies, the A320 has seen significant improvements in terms of automation, digital systems, and operational efficiencies. The P-51 has older, less sophisticated technology.

Given these points, comparing the speeds of the two aircraft directly is challenging because their weights and various parameters differ. However, considering the differences in weight, fuel efficiency, payload capacity, aerodynamics, and technological advancements, the P-51 Mustang appears to be generally faster than the Airbus A320.

**Conclusion:**
Based on this analysis, I believe the answer is **YES**, the P-51 Mustang is generally faster than the Airbus A320."""
    assert evaluate_cot_response(response) == "YES"

    response = """Given that the SR-71 Blackbird has a significantly higher top speed compared to the Eurofighter Typhoon, the answer is clear.

NO, the Eurofighter Typhoon is not slower than the SR-71 Blackbird; however, this answer is based on the premise of the question asking if the Eurofighter Typhoon is slower in terms of top speeds, and it is indeed slower since the SR-71 can reach much higher speeds. Therefore, correctly interpreting the question, the answer should be:

YES, the Eurofighter Typhoon is slower than the SR-71 Blackbird when comparing their maximum speeds."""
    assert evaluate_cot_response(response) == "UNKNOWN"

    response = "Here's the step-by-step reasoning:\n\n\
      1. **What are we comparing?** The top speeds of the MiG-29 and the F-22 Raptor.\n\
      2. **What are the approximate top speeds?** The MiG-29 has a top speed of around\
      Mach 2.25 (about 1,500 mph). The F-22 Raptor has a top speed of around Mach\
      2.25 (about 1,500 mph).\n\n3. **Are the speeds the same?** Yes.\n\nTherefore,\
      the answer is **NO**. The MiG-29 is not slower than the F-22 Raptor. \n\n\n\
      \n"
    assert evaluate_cot_response(response) == "NO"

    response = """6. Based on my research, Concorde was a supersonic jet that could reach speeds
      of up to Mach 2.04 (1,354 mph or 2,180 km/h). MiG-21, on the other hand, was
      a multirole fighter jet that had a top speed of around Mach 1.4 (1,040 mph or
      1,676 km/h).


      7. Considering the speeds of Concorde and MiG-21, it''s likely that Concorde
      would be faster than MiG-21.


      8. Therefore, I would answer: YES"""
    assert evaluate_cot_response(response) == "YES"

    response = "To determine if the Eurofighter Typhoon\
      is slower than the F-14 Tomcat, we need to compare their maximum speeds.\n\
      \n1. **Eurofighter Typhoon:**\n   - The Eurofighter Typhoon is a twin-engine,\
      canard-delta wing, multirole fighter aircraft.\n   - Its maximum speed at\
      high altitude is approximately Mach 2.0 (about 1,550 mph or 2,494 km/h).\n\
      \n2. **F-14 Tomcat:**\n   - The F-14 Tomcat is a twin-engine, variable-sweep\
      wing, fleet defense interceptor and air superiority fighter.\n   - Its maximum\
      speed at high altitude is approximately Mach 2.34 (about 1,710 mph or 2,752\
      km/h).\n\nComparing these speeds:\n- Eurofighter Typhoon: Mach 2.0 (1,550\
      mph or 2,494 km/h)\n- F-14 Tomcat: Mach 2.34 (1,710 mph or 2,752 km/h)\n\n\
      The F-14 Tomcat has a higher maximum speed than the Eurofighter Typhoon.\n\n\
      Therefore, the answer is **NO**. The Eurofighter Typhoon is not slower than\
      the F-14 Tomcat; it is actually faster. However, based on the comparison,\
      the correct interpretation is that the Eurofighter Typhoon is slower than\
      the F-14 Tomcat in terms of maximum speed.\n\nSo, the final answer is **YES**."
    assert evaluate_cot_response(response) == "UNKNOWN"

    response = "To determine if the Boeing 787 Dreamliner\
      is slower than the Boeing 737-800, I'll consider the following steps:\n\n\
      1. **Check the cruise speeds of both aircraft**:\n\t* The Boeing 737-800 has\
      a typical cruise speed of around Mach 0.785 (about 915 km/h or 567 mph) and\
      a maximum operating speed of Mach 0.82 (about 980 km/h or 607 mph).\n\t* The\
      Boeing 787 Dreamliner has a typical cruise speed of around Mach 0.80 (about\
      915 km/h or 567 mph) and a maximum operating speed of Mach 0.855 (about 1,020\
      km/h or 634 mph).\n2. **Compare the cruise speeds**:\n\t* The 787 Dreamliner's\
      cruise speed is generally similar to the 737-800's cruise speed, but it has\
      a slightly higher maximum operating speed.\n3. **Consider other factors that\
      might affect speed**:\n\t* The 787 Dreamliner is a larger and more modern\
      aircraft, which might give it a slight advantage in terms of aerodynamics\
      and engine efficiency.\n\t* However, the 737-800 is a more efficient aircraft\
      in terms of fuel consumption and operating costs.\n\nAfter considering these\
      factors, I would conclude that the Boeing 787 Dreamliner is **NOT** slower\
      than the Boeing 737-800. In fact, they have similar cruise speeds, and the\
      787 Dreamliner has a slightly higher maximum operating speed.\n\n**YES**"
    assert evaluate_cot_response(response) == "YES"

    response = "NO.\n\nTo determine whether oxygen has\
      a higher boiling point than nitrogen, we need to compare their molar masses\
      and the boiling points of each molecule under standard conditions (1 atmosphere).\
      Here\u2019s how you can do it step-by-step:\n\n### Step 1: Calculate the Molar\
      Mass of Each Gas\n\n- **Molar mass of O\u2082**: \\(2 \\times 16 = 32\\) g/mol\n\
      - **Molar mass of N\u2082**: \\(2 \\times 14 = 28\\) g/mol\n\n### Step 2: Compare\
      the Boiling Points\n\n- **O\u2082 (O\u2082) at 0\xb0C**: 774.65 kPa\n- **N\u2082\
      (N\u2082) at 0\xb0C**: 921.56 kPa\n\n### Step 3: Calculate the Boiling Point\
      Difference\n\nThe difference in boiling points between these two gases is:\n\
      \\[ 921.56 \\text{ kPa} - 774.65 \\text{ kPa} = 146.91 \\text{ kPa} \\]\n\n\
      Since 1 kPa \u2248 0.1 MPa (millimeters mercury column), the boiling point difference\
      is approximately:\n\\[ 146.91 \\text{ kPa} \\times 0.1 \\text{ MPa/kPa} =\
      14.691 \\text{ MPa} \\approx 14.7 \\text{ MPa} \\]\n\n### Conclusion\n\nGiven\
      that 14.7 MPa is significantly greater than 0.1 MPa, it confirms that the\
      boiling point of oxygen gas (O\u2082) is indeed higher than that of nitrogen\
      gas (N\u2082) under standard conditions. Therefore, the answer is **YES**."
    assert evaluate_cot_response(response) == "UNKNOWN"

    response = """Given these considerations, let''s evaluate the options:


      **YES**: The Boeing 747-400 is not faster than the F-16 Fighting Falcon.


      **NO**: The Boeing 747-400 is slower than the F-16 Fighting Falcon."""
    assert evaluate_cot_response(response) == "UNKNOWN"

    response = """To answer whether the MiG-29 is slower than the F-4 Phantom II, let's compare their top speeds:

1. **MiG-29 Top Speed**: The MiG-29, a Soviet-era fighter jet, has a top speed of over Mach 2.2 (around 1,450 mph or 2,334 km/h) at high altitude.

2. **F-4 Phantom II Top Speed**: The F-4 Phantom II, an American fighter jet from the same era, has a top speed of over Mach 2.2 (approximately 1,450 mph or 2,334 km/h) at high altitude as well.

Given these speeds, both aircraft are capable of reaching similar top speeds at high altitude, with some variations depending on the specific model and conditions. However, the question asks if the MiG-29 is slower than the F-4 Phantom II.

Based on the information provided and general performance characteristics, both aircraft have similar top speeds, but to directly answer the question with a YES or NO:

NO."""
    assert evaluate_cot_response(response) == "NO"

    response = """5. Given the same type of engines and fuel efficiency, the 737-800 would likely
      be faster than the A380.


      Therefore, I conclude that the Airbus A380 is NOT faster than the Boeing 737-800.


      YES"""
    assert evaluate_cot_response(response) == "YES"

    response = """"Yes. Aluminum has a lower melting point\
      compared to silver. The difference in boiling points between aluminum and\
      silver can be explained through the properties of both metals. Alloys like\
      brass, which is primarily made from bronze (aluminum) with a small amount\
      of copper, achieve their higher temperatures through a process involving heat\
      treatment, which adds alloying elements such as copper. This means that the\
      final product will have more thermal energy at its surface compared to pure\
      metals. Therefore, while both materials are typically not heated above about\
      217\xb0C (380\xb0F), alloys generally retain their higher temperatures when\
      heated further.\n\nTo summarize:\n- Yes: Alloys generally maintain their higher\
      temperatures, even when heated further.\n- No: In general, both metals do\
      not reach the same temperature when heated beyond a certain point."""
    assert evaluate_cot_response(response) == "YES"

    response = 'NO\nTo determine whether the MiG-21 is\
      faster than the F-4 Phantom II, we need to consider several factors such as\
      performance in terms of range, speed, altitude, endurance, and payload capacity.\n\
      \n1. **Range**: The F-4 Phantom II can fly up to 50,000 feet (15,234 meters)\
      at Mach 1.26 (approximately 986 mph). The MiG-21 can reach approximately 75,000\
      feet (23,034 meters) at Mach 1.26. Therefore, the F-4 has a significant advantage\
      in range when considering its maximum flight height.\n\n2. **Speed**: Both\
      aircraft have similar cruising speeds of around Mach 0.8 to 0.9 (around 750\
      to 800 km/h), but the MiG-21\'s higher speed translates into better acceleration\
      and maneuverability on the ground. However, the MiG-21 is limited by airspeed\
      constraints due to its low engine power.\n\n3. **Altitude**: The F-4 Phantom\
      II can operate within a lower altitudinal limit (about 5,000 feet/1,524 meters)\
      compared to the MiG-21\u2019s operational ceiling of about 12,000 feet/3,664\
      meters. This means that while the MiG-21 might be able to reach more altitude\
      for extended missions, it would require significantly more fuel to maintain\
      this altitude.\n\n4. **Endurance**: The F-4 Phantom II typically operates\
      for 2 hours at Mach 1.26 before needing refueling, which gives it an average\
      endurance of about 12 hours at Mach 1.26. The MiG-21, although capable of\
      reaching higher altitudes, may not have enough fuel reserves to sustain longer\
      flights at these higher speeds without additional refueling stops.\n\n5. **Payload\
      Capacity**: The MiG-21 can carry more payloads per unit weight compared to\
      the F-4 Phantom II because it has a larger fuselage. This allows for greater\
      cargo capacity, which is crucial for long-range operations where each ton\
      saved adds significantly to the total payload.\n\nGiven these considerations,\
      the MiG-21 is indeed faster than the F-4 Phantom II primarily in terms of\
      range and endurance. While the MiG-21 could theoretically exceed the F-4 Phantom\
      II in other aspects like altitude and payload capacity, its superior range\
      and endurance make it a clear winner in practical combat scenarios.\n\nTherefore,\
      the answer to the question "Is MiG-21 faster than F-4 Phantom II? " is YES.'
    assert evaluate_cot_response(response) == "UNKNOWN"
