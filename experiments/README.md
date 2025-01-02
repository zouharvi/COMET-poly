|logname|description|
|-|-|
|firstrun|just 5% of the data, not really useful, PW>>DA on pairwise|
|secondrun|10% of the data, PW slightly worse than DA on pairwise but much better on ranking|
|pw_dedup|save every 10k steps, deduplicate comparisons, 3_000 frozen steps, TODO: in ranking might need to move the threshold from 0.5 to median or 0.55 because we'll have fewer zeros|