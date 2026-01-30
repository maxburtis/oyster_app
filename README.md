🦪 Oyster Calculator — README (Sampling Protocol + How to Run the App)

This README covers (1) a field protocol for sampling grow-out bags and (2) how to run and use the Streamlit app to generate growth forecasts, % market-ready, and bag split timing.

Reference protocol details (biweekly sampling, 200-oyster bag, wash/weigh method, mortality definition, market density sampling) are adapted from your grant write-up.  ￼

⸻

1) Sampling protocol (bags of oysters)

Sampling cadence + design
	•	Frequency: every 2 weeks during the grow-out season.  ￼
	•	Per site: maintain a designated “sample bag” placed near the same location each time (e.g., adjacent to buoy / representative cage).  ￼
	•	Initial sample size: target ~200 oysters per sample bag at the start; continue sampling the same bag over time, adjusting counts as mortalities occur.  ￼

Optional (for market density calibration):
	•	Graduated container + water for displacement volume (done once per season / batch).  ￼

⸻

A. Biweekly sampling steps (weight + mortality)
	1.	Retrieve the sample bag
	    •	Bring the sample bag to the processing float / work area.
	2.	Clean
	    •	Remove oysters from the bag and wash off fouling and pseudofeces thoroughly.  ￼
	3.	Assess mortality
        •	Count how many are dead.
        •	Mortality rule: any oyster with an open shell is considered dead.  ￼
        •	Discard dead oysters.
	4.	Count live oysters
	    •	Record the live Count remaining in the bag after removing dead.
	5.	Weigh for total live bag weight
        •	Tare the scale using the empty grow-out bag (or record tare weight once and reuse).
        •	Put all live oysters back into the bag and weigh.
        •	Record:
        •	weight_g = total live oyster mass (grams)
        •	Count = number of live oysters
	6.	Compute average weight
	    •	Avg_Weight_g = weight_g / Count
	7.	Return to grow-out
	    •	Put oysters back into their gear location until the next sampling interval.

B. One-time “market oyster” sampling (for density calibration)

Do this once per season (or whenever gear/size class changes materially):
	1.	Randomly sample ~50 market-ready oysters.
	2.	Measure total mass (same method as above).
	3.	Measure volume using water displacement in a graduated container.
	4.	Compute density: Density (g/L) = total_mass_g / total_volume_L
That density is what the app uses to convert biomass → liters for split planning.  ￼

2) CSV format required by the app

The app expects a CSV upload with at minimum:

Required columns
	•	Date (parseable date)
	•	Bag (bag identifier)
    •	Count (enables biomass/volume and split timing)
	•	AND EITHER:
	•	Avg_Weight_g
	•	or both weight_g and Count (the app will compute Avg_Weight_g)

Example CSV

Date,Bag,Count,weight_g
2026-04-15,MBUR520_A,200,4200
2026-04-29,MBUR520_A,198,4554
2026-05-13,MBUR520_A,198,4980


3) Using the app (what each input means)

Upload
	•	Use the sidebar “Upload bio CSV” and select your sampling CSV.

Growth model settings
	•	Weight CV (CV): controls the spread of sizes around the mean (used to estimate what fraction exceed market size).
	•	Market size (g) (MARKET_WEIGHT): the threshold that defines “market-ready.” The default is 66g which equates to an average 3" oyster. 
	•	Months to project: how far beyond your last observation to forecast.

    Coefficient of Variation (CV) describes how variable individual oyster sizes are within a bag. A low CV means oysters are very uniform; a high CV means a wide spread of sizes. The app assumes individual oyster weights follow a lognormal distribution with the entered CV, which is how it estimates the percent of oysters at or above market size. Even if the average oyster is market size, a high CV means many individuals may still be undersized.

    Density (g/L) converts total oyster biomass into bag volume and is critical for predicting bag fullness and split timing. In the SARE grant work, density is measured empirically by weighing a known volume of oysters (e.g., filling a container or bag section of known liters, then weighing the oysters in grams). Density is calculated as total grams divided by liters occupied. This approach captures real-world packing differences due to gear type, oyster shape, and handling practices, making split projections more realistic.

    Split factor defines how full a bag is allowed to get before a split is recommended, expressed as a multiple of the initial bag volume. For example, with a 3 L starting bag and a split factor of 2.0, the app flags a split when volume reaches 6 L per bag. Lower split factors produce more frequent, conservative splits; higher values allow tighter packing and fewer splits. Together, CV, density, and split factor don’t change growth itself — they translate biological growth into practical harvest and labor decisions.


5) Outputs you get

After upload, the app produces:
	•	Plot 1: observed weights + predicted mean weight over time
	•	Plot 2: predicted % at/above market over time
	•	Plot 3: projected volume and split threshold, with split event markers
	•	A Download forecast CSV button that exports daily projections per bag (oyster_market_forecast.csv)


6) Troubleshooting

“CSV must contain Avg_Weight_g OR (weight_g + Count)”
	•	Add either Avg_Weight_g, or both weight_g and Count.

Volume/split schedule is disabled
	•	Your CSV is missing Count (or has blanks). Add Count for each bag/date.

Weird jumps or flat forecasts
	•	Usually caused by inconsistent Bag IDs, missing dates, or very few data points per bag.
	•	Make sure each bag has multiple biweekly samples across the season.

