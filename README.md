🦪 Oyster Calculator — README (Sampling Protocol + How to Run the App)

This README covers (1) a field protocol for sampling grow-out bags and (2) how to run and use the Streamlit app to generate growth forecasts, % market-ready, and bag split timing.

Reference protocol details (biweekly sampling, 200-oyster bag, wash/weigh method, mortality definition, market density sampling) are adapted from your grant write-up.  ￼

⸻

1) Sampling protocol (bags of oysters)

Goals
	•	Track growth (average weight) and mortality through the season.
	•	Generate a dataset you can upload to the app to forecast:
	•	mean weight vs date
	•	% of oysters above market size
	•	biomass → bag volume → split schedule

Sampling cadence + design
	•	Frequency: every 2 weeks during the grow-out season.  ￼
	•	Per site: maintain a designated “sample bag” placed near the same location each time (e.g., adjacent to buoy / representative cage).  ￼
	•	Initial sample size: target ~200 oysters per sample bag at the start; continue sampling the same bag over time, adjusting counts as mortalities occur.  ￼

Equipment checklist
	•	Grow-out bag (the sample bag)
	•	Scale that can hang a full bag (crane/fish scale works well)
	•	High-pressure washdown hose
	•	Bucket / tote for handling
	•	Data sheet (or phone form) for logging Bag ID, Date, counts, weight

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

Time expectation: the write-up estimates ~3 hours for one person per biweekly sampling session (per site), depending on workflow.  ￼

⸻

B. One-time “market oyster” sampling (for density calibration)

Do this once per season (or whenever gear/size class changes materially):
	1.	Randomly sample ~200 market-ready oysters.
	2.	Measure total mass (same method as above).
	3.	Measure volume using water displacement in a graduated container.
	4.	Compute density:

	•	Density (g/L) = total_mass_g / total_volume_L

That density is what the app uses to convert biomass → liters for split planning.  ￼

⸻

Data quality rules (important)
	•	Consistent Bag IDs over time (exact spelling/case).
	•	Always log Date (same timezone; preferably the day you sampled).
	•	Always log Count if you want volume + split schedule to work.
	•	If you ever weigh only a subsample (not the whole bag), note it clearly—otherwise the model will treat it like full-bag data.

⸻

2) CSV format required by the app

The app expects a CSV upload with at minimum:

Required columns
	•	Date (parseable date)
	•	Bag (bag identifier)
	•	AND EITHER:
	•	Avg_Weight_g
	•	or both weight_g and Count (the app will compute Avg_Weight_g)

Strongly recommended
	•	Count (enables biomass/volume and split timing)

Example CSV

Date,Bag,Count,weight_g
2026-04-15,MBUR520_A,200,4200
2026-04-29,MBUR520_A,198,4554
2026-05-13,MBUR520_A,198,4980

Or if you already computed average weights:

Date,Bag,Count,Avg_Weight_g
2026-04-15,MBUR520_A,200,21.0
2026-04-29,MBUR520_A,198,23.0


⸻

3) How to run the app (Streamlit)

A. Install dependencies

From your project folder (where app.py lives):

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install streamlit pandas numpy pygam scipy matplotlib

B. Run Streamlit

streamlit run app.py

Streamlit will print a local URL (usually http://localhost:8501). Open it in your browser.

⸻

4) Using the app (what each input means)

Upload
	•	Use the sidebar “Upload bio CSV” and select your sampling CSV.

Growth model settings
	•	Weight CV (CV): controls the spread of sizes around the mean (used to estimate what fraction exceed market size).
	•	Market size (g) (MARKET_WEIGHT): the threshold that defines “market-ready.”
	•	Months to project: how far beyond your last observation to forecast.

Bagging / splits

These control split timing (volume per bag):
	•	Density (g/L): converts biomass to liters. Use your measured density from the one-time market sample when possible.
	•	Split factor (x initial volume): how much the bag can “grow” (in volume) before you split.
	•	Initial bag volume (L): starting “allowed” volume per bag.

Note: split schedule only runs if your CSV includes Count.

⸻

5) Outputs you get

After upload, the app produces:
	•	Plot 1: observed weights + predicted mean weight over time
	•	Plot 2: predicted % at/above market over time
	•	Plot 3: projected volume and split threshold, with split event markers
	•	A Download forecast CSV button that exports daily projections per bag (oyster_market_forecast.csv)

⸻

6) Troubleshooting

“CSV must contain Avg_Weight_g OR (weight_g + Count)”
	•	Add either Avg_Weight_g, or both weight_g and Count.

Volume/split schedule is disabled
	•	Your CSV is missing Count (or has blanks). Add Count for each bag/date.

Weird jumps or flat forecasts
	•	Usually caused by inconsistent Bag IDs, missing dates, or very few data points per bag.
	•	Make sure each bag has multiple biweekly samples across the season.

⸻
