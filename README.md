# 🦪 Oyster Calculator  
### Sampling Protocol + How to Run the App

This app uses biweekly sampling data from oyster grow-out bags to project growth, estimate market readiness, and plan bag splits. It is designed to reflect real farm practices and was developed as part of a **SARE (Sustainable Agriculture Research & Education)**–funded project.

---

## 1) Sampling protocol (bags of oysters)

### Sampling cadence and design
- **Frequency:** every 2 weeks during the grow-out season
- **Per site:** maintain a designated *sample bag* placed near the same location each time (e.g., adjacent to a buoy or representative cage)
- **Initial sample size:** target ~200 oysters per sample bag at the start; continue sampling the same bag over time, adjusting counts as mortalities occur

**Optional (for market density calibration):**
- Graduated container and water for displacement volume (done once per season or batch)

---

### A. Biweekly sampling steps (weight + mortality)

1. **Retrieve the sample bag**
   - Bring the sample bag to the processing float or work area
2. **Clean**
   - Remove oysters from the bag and wash off fouling and pseudofeces thoroughly
3. **Assess mortality**
   - Count how many oysters are dead
   - **Mortality rule:** any oyster with an open shell is considered dead
   - Discard dead oysters
4. **Count live oysters**
   - Record the live **Count** remaining in the bag
5. **Weigh total live bag weight**
   - Tare the scale using the empty grow-out bag (or record tare weight once and reuse)
   - Put all live oysters back into the bag and weigh
   - Record:
     - `weight_g` = total live oyster mass (grams)
     - `Count` = number of live oysters
6. **Compute average weight**
   - `Avg_Weight_g = weight_g / Count`
7. **Return to grow-out**
   - Put oysters back into their gear location until the next sampling interval

---

### B. One-time “market oyster” sampling (density calibration)

Do this once per season (or whenever gear or size class changes materially):

1. Randomly sample ~50 market-ready oysters
2. Measure total mass (same method as above)
3. Measure volume using water displacement in a graduated container
4. Compute density:

```
Density (g/L) = total_mass_g / total_volume_L
```

This density is used by the app to convert biomass into liters for bag volume and split planning.

---

## 2) CSV format required by the app

### Required columns
- `Date` (parseable date, e.g., YYYY-MM-DD)
- `Bag` (bag identifier)
- `Count` (enables biomass, volume, and split timing)

### And either
- `Avg_Weight_g`, or
- both `weight_g` and `Count` (the app will compute `Avg_Weight_g`)

### Example CSV
```csv
Date,Bag,Count,weight_g
2026-04-15,MBUR520_A,200,4200
2026-04-29,MBUR520_A,198,4554
2026-05-13,MBUR520_A,198,4980
```

---

## 3) Using the app

### Upload
- Use the sidebar **“Upload bio CSV”** and select your sampling file

### Growth model settings
- **Weight CV (Coefficient of Variation):** controls how variable individual oyster sizes are within a bag
- **Market size (g):** threshold defining “market-ready” (default 66 g ≈ average 3" oyster)
- **Months to project:** how far beyond your last observation to forecast

---

## 4) Key model inputs explained

### Coefficient of Variation (CV)
CV describes how variable individual oyster sizes are within a bag.

```
CV = standard deviation of individual weights / mean individual weight
```

Low CV means oysters are very uniform; high CV means a wide spread of sizes. The app assumes individual oyster weights follow a lognormal distribution with the specified CV to estimate the **percent of oysters at or above market size**. Even if the average oyster is market size, a high CV means many individuals may still be undersized.

Typical values:
- 0.20–0.25 → very uniform crop
- 0.30–0.35 → common farm variability
- >0.40 → highly uneven sizes

---

### Density (g/L)
Density converts total oyster biomass into bag volume:

```
Volume (L) = total biomass (g) / density (g/L)
```

In the SARE grant work, density is measured empirically by weighing a known volume of oysters (for example, filling a container or bag section of known liters and weighing the oysters). This captures real packing differences due to gear type, oyster shape, and handling practices.

Density is used to project bag fullness and determine split timing.

---

### Split factor
The split factor defines how full a bag is allowed to get before a split is recommended. It is expressed as a multiple of the initial bag volume.

Example:
- Initial bag volume = 3.0 L
- Split factor = 2.0
- Split threshold = 6.0 L per bag

Lower split factors produce more frequent, conservative splits; higher values allow tighter packing and fewer splits. CV, density, and split factor do not change growth itself — they translate growth into **practical harvest and labor decisions**.

---

## 5) Outputs

After upload, the app produces:
- Plot 1: observed weights and predicted mean weight over time
- Plot 2: predicted percent at or above market size over time
- Plot 3: projected bag volume with split threshold and split events
- Downloadable forecast CSV (`oyster_market_forecast.csv`) with daily projections per bag

---

## 6) Troubleshooting

### “CSV must contain Avg_Weight_g OR (weight_g + Count)”
- Add either `Avg_Weight_g`, or both `weight_g` and `Count`

### Volume or split schedule disabled
- Your CSV is missing `Count` or contains blank values

### Weird jumps or flat forecasts
- Usually caused by inconsistent Bag IDs, missing dates, or very few data points per bag
- Ensure each bag has multiple biweekly samples across the season

---

## Credits
This work was funded by a **SARE (Sustainable Agriculture Research & Education)** grant.
