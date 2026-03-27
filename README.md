# Project Overview

Analysis and simulation of water quality in the Nancy region, focusing on industrial chloride discharges and environmental stressors.

### Chloride Discharge Data [1][2][3]
**The Crucial Role of Agricultural Crowdsourcing:**
This application enables farmers and local residents to report exceptional events (e.g., fertilizer spreading, untreated water leaks). **Inputting this data is vital:** fertilizers and chemicals that run off into rivers eventually infiltrate adjacent groundwater aquifers. Pumping this contaminated water for irrigation directly impacts local crop quality, agricultural yields, and ultimately, the agricultural economy. Reporting these events early allows for the anticipation of contamination waves before they reach critical thresholds that would harm the farming community itself.

### Chloride Discharge Data

- **Annual total**: ~1,000,000 tonnes discharged into the Meurthe river downstream of local soda plants.
- **Solvay (Dombasle)**: 16.25 kg/s average annual flux.
- **Novacarb (Laneuveville)**: 13.48 kg/s average annual flux.
- **Combined limit**: 31 kg/s (Bonn Convention limit for both sites).
- **Reduction target**: 15% reduction objective via new sodium chloride recycling processes.

### Seasonal Environmental Risks [1][2][3]

- **Winter**: Snow melting increases water volume and dilution but brings high chloride and heavy metal loads from road maintenance salt.
- **Summer**: Drought periods reduce water volume, decreasing dilution capacity and intensifying industrial effluent concentration.
- **Heavy Rain**: Urban runoff and sewer overflows increase turbidity, mudslides, and stagnant nitrate concentrations.
- **Planting Period**: Fertilizer application followed by rain causes nitrogen runoff, significantly increasing nitrates and eutrophication risk.

### Correlations and Remarks
- **Photosynthesis**: High salt concentration and turbidity reduce light penetration, decreasing photosynthesis and dissolved oxygen.
- **Dilution**: Water levels are inversely proportional to pollutant concentration for a constant industrial discharge.
- **Ecosystem Stress**: Hypoxia (O2 < 50%) is lethal for sensitive species like salmonids present in the Grand Est.

### Stakeholder Indicators (France & Grand Est Context)
Numerical thresholds derived from DRIEAT Grand Est, Agence de l'Eau Rhin-Meuse, and French Public Health Code [1][5][6].

#### Indicators for Farmers (Irrigation) [4]
- **Salinity (Conductivity)**:
  - `< 0.7 mS/cm`: Safe for all regional crops.
  - `0.7 – 3.0 mS/cm`: Moderate stress; yield reduction risk for sensitive crops.
  - `> 3.0 mS/cm`: High salinity; serious yield loss expected.
- **Nitrates (NO3)**:
  - `< 25 mg/L`: Good status; minor nitrogen contribution from water.
  - `> 50 mg/L`: High risk of over-fertilization and regulatory leaching.
- **Algal Risk (Chlorophyll-a)**:
  - `> 10 µg/L`: Risk of biofilm and irrigation emitter clogging.
- **Physical Load (Turbidity)**:
  - `< 20 NTU`: Safe for drip-irrigation.
  - `> 50 NTU`: High clogging risk; specialized filtration required.

#### Indicators for Local Ecosystems (DCE Standards) [5]
- **Oxygen Saturation (O2 %)**:
  - `> 80 %`: Excellent status for Meurthe river biodiversity.
  - `50 – 70 %`: Moderate stress for salmonids and macro-invertebrates.
  - `< 50 %`: Hypoxia risk; lethal for fish and invertebrates.
- **Nitrates (NO3)**:
  - `< 25 mg/L`: Good environmental status threshold.
  - `> 50 mg/L`: Poor status; primary driver for eutrophication and algal blooms.
- **pH**:
  - `6.5 – 8.2`: Optimal range for regional aquatic life.
- **Total Suspended Solids (MES)**:
  - `< 25 mg/L`: Clear water favoring healthy fish gill function and spawning grounds.

#### Indicators for Leisure and Drinking Water [1][3][6]
- **Swimming (Recreational)**:
  - **Turbidity**: `< 50 NTU` (transparency limit for safety and visibility).
  - **pH**: `7.0 – 8.0` (optimal to prevent skin and eye irritation).
  - **Chlorophyll-a**: `< 10 µg/L` (low risk of cyanotoxin exposure).
- **Drinking Water (AEP)**:
  - **Nitrates**: `< 50 mg/L` (French Public Health Code regulatory limit).
  - **Chlorides**: `< 250 mg/L` (Guideline for taste and distribution network protection).
  - **Turbidity**: `< 1.0 NTU` (Target for treatment efficiency at plant exit).
  - **Conductivity**: `< 1100 µS/cm` (Standard for non-brassy, high-quality potable water).

### Sources

- [1] [Grand Est DRIEAT - Prélèvements et rejets](https://www.grand-est.developpement-durable.gouv.fr/prelevements-pressions-et-rejets-a12452.html?lang=fr)
- [2] [Eau Rhin-Meuse - Note enjeu département 54](https://www.eau-rhin-meuse.fr/sites/default/files/2025-09/CNE_Note_enjeu_d%C3%A9partement_54_mai2025_2.pdf)
- [3] [Région Grand Est - Diagnostic Eau](https://www.grandest.fr/wp-content/uploads/2019/07/piece-n07-annexe-6-diagnostic-eau.pdf)
- [4] [FAO - Water Quality for Agriculture](https://www.fao.org/3/t0566e/t0566e05.htm)
- [5] [Three Main Types of Water Quality Parameters: A Comprehensive Guide](https://e.yosemitech.com/industry/Three-Main-Types-of-Water-Quality-Parameters.html)
- [6] [WHO & French Public Health Code - Guidelines for Drinking-Water Quality](https://iris.who.int/server/api/core/bitstreams/b437749a-43f7-472f-a6d9-42596e8ac0ae/content)
