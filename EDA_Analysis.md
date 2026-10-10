# EDA Analysis

## Sources and method

This report documents how the EDA findings were obtained from:

- `AIML_Project_1_Full_Code_Notebook.ipynb`: reviewed the data overview and EDA cells, including plot code, interpretation comments, and saved text outputs.
- `wind_turbine_detection.csv`: independently calculated summary statistics using pandas in the terminal.

The original notebook and CSV were not modified. The numerical conclusions below come from calculations on the CSV, rather than estimates read from plots. In this dataset, `failure = 0` means normal and `failure = 1` means fault. Each row is an observation; multiple fault-labeled observations can belong to the same fault episode.

## 1. Fault observations are rare

**Calculation:** Count each value of `failure` and divide by the total number of rows.

| Class | Records | Percentage |
|---|---:|---:|
| Normal | 127,807 | 97.00% |
| Fault | 3,953 | 3.00% |
| Total | 131,760 | 100.00% |

**Conclusion:** The target is strongly imbalanced. Predicting normal for every observation would achieve approximately 97% accuracy while detecting no faults. Model evaluation should therefore examine fault recall and precision, alongside the confusion matrix. These percentages describe observations, not the percentage of turbines that broke down.

## 2. Vibration and oil particles show the clearest individual fault associations

**Calculation:** Group records by `failure` and compare sensor medians. Also calculate Pearson correlations between numerical features and the binary target.

| Measurement | Normal median | Fault median | Correlation with failure |
|---|---:|---:|---:|
| Drivetrain vibration RMS (mm/s) | 1.681 | 3.635 | 0.537 |
| Oil particle count | 64.450 | 165.209 | 0.473 |
| Gearbox bearing temperature (degrees C) | 57.935 | 69.072 | 0.100 |

Drivetrain vibration and oil particle count had the largest positive Pearson correlations with the target among the numerical input columns. Their fault-class medians are approximately 2.16 and 2.56 times their normal-class medians, respectively.

**Conclusion:** These are promising inputs for fault detection. Gearbox bearing temperature also differs between classes, but its overall linear association is weaker. Differences in medians do not establish a reliable classification threshold, causation, or performance on unseen data. Combining measurements is a modeling hypothesis that needs validation.

## 3. Power output needs operating context

**Calculation:** Reproduce the notebook's ten equal-width wind-speed bins and calculate average power output in each bin. Compare power output with the fault label.

| Wind-speed interval (m/s, rounded) | Mean power output (kW) |
|---|---:|
| Approximately 0 to 3.214 | 1.11 |
| 3.214 to 6.412 | 37.38 |
| 6.412 to 9.611 | 396.41 |
| 9.611 to 12.809 | 1,552.16 |
| 12.809 to 16.008 | 2,181.42 |
| 16.008 to 19.206 | 2,202.22 |
| 19.206 to 22.405 | 2,252.18 |
| 22.405 to 25.603 | 2,019.05 |
| 25.603 to 28.802 | 0.00 |
| 28.802 to 32.000 | 0.00 |

The Pearson correlation between power output and `failure` is approximately -0.020. The notebook also notes substantial overlap in the power distributions across classes.

**Conclusion:** Power rises with wind speed, levels off, and is zero in the two highest bins. This pattern is consistent with an operating limit, but the grouped averages alone do not prove the cause. Low output alone is insufficient evidence of a fault. These averages combine turbines with different rated capacities, and the highest bins contain few observations.

## 4. Fault prevalence differs across turbines

**Calculation:** Group by `turbine_id` and take the mean of the binary `failure` column. Multiply by 100 to obtain the percentage of observations labeled as faults.

| Turbine | Fault-labeled observations |
|---|---:|
| T001 | 6.35% |
| T007 | 5.60% |
| T008 | 5.44% |
| T014 | 0.97% |

T001 has the highest observed fault proportion and T014 the lowest.

**Conclusion:** Asset-level differences warrant investigation. These proportions do not establish why turbines differ or how many separate breakdowns occurred. The notebook's scatter plot of turbine ID versus a binary label cannot reliably show these proportions because points overlap; grouped rates provide the evidence.

## 5. Rotor and generator speed contain highly overlapping information

**Calculation:** Calculate the Pearson correlation between `rotor_speed_rpm` and `generator_speed_rpm`.

**Result:** 0.99984, an almost perfect positive linear relationship.

**Conclusion:** The two measurements contain substantially redundant information. This is relevant to feature selection and interpretation, but it does not automatically require removing a feature or prove that the drivetrain is healthy.

## 6. Wind speeds are right-skewed

**Calculation:** Calculate quantiles, mean, median, range, and sample skewness for `wind_speed_mps`.

| Statistic | Value |
|---|---:|
| Mean | 7.65 m/s |
| Median | 7.14 m/s |
| 25th percentile | 4.63 m/s |
| 75th percentile | 10.09 m/s |
| 95th percentile | 15.13 m/s |
| Maximum | 32.00 m/s |
| Sample skewness | 0.778 |

**Conclusion:** Low to moderate speeds are common, with a tail toward high speeds. The middle half of observations lies between 4.63 and 10.09 m/s. High values are uncommon but are not automatically errors. The mean alone does not establish what percentage of observations falls below it.

## Corrections to existing notebook comments

The following means were recalculated directly from the CSV:

| Feature | Approximate mean stated in notebook comments | Actual mean |
|---|---:|---:|
| Drivetrain vibration RMS | 0.5 mm/s | 1.835 mm/s |
| Tower vibration | 0.3 mm/s | 1.135 mm/s |
| Oil particle count | 1,500 | 73.727 |

These comments should be corrected before presenting the notebook. Also, the central line in a standard seaborn box plot represents the median, not the mean. The original notebook has not been edited as part of this report.

## Reproduce the calculations

Run the following code in a notebook cell or Python script with the working directory set to this project folder. It requires pandas and prints the statistics without changing the dataset.

```python
from pathlib import Path
import pandas as pd

data = pd.read_csv(Path('wind_turbine_detection.csv'))

# 1. Class counts, percentages, and all-normal accuracy baseline.
counts = data['failure'].value_counts().sort_index()
print('\nClass distribution:')
print(pd.DataFrame({
    'records': counts,
    'percentage': counts / len(data) * 100,
}))
print('All-normal accuracy:', (data['failure'] == 0).mean())

# 2. Compare medians and rank individual linear associations.
features = [
    'drivetrain_vibration_rms_mmps',
    'oil_particle_count',
    'gearbox_bearing_temp_C',
]
medians = data.groupby('failure')[features].median().T
print('\nMedians by class (0 = normal, 1 = fault):')
print(medians)
print('\nFault / normal median ratios:')
print(medians[1] / medians[0])
correlations = data.select_dtypes(include='number').corr()['failure']
print('\nFeature correlations with failure:')
print(correlations.drop('failure').sort_values(ascending=False))

# 3. Match the notebook's ten equal-width wind-speed bins.
# Include counts to show how much data supports each mean.
wind_bins = pd.cut(data['wind_speed_mps'], bins=10)
print('\nPower output by wind-speed interval:')
print(data.groupby(wind_bins, observed=False)['power_output_kW']
      .agg(['count', 'mean']))
print('\nPower output quartiles by class:')
print(data.groupby('failure')['power_output_kW']
      .quantile([0.25, 0.5, 0.75]).unstack())

# 4. Observation-level fault proportions, not separate fault events.
turbine_summary = data.groupby('turbine_id')['failure'].agg(
    records='size', fault_records='sum', fault_fraction='mean'
)
turbine_summary['fault_percentage'] = turbine_summary['fault_fraction'] * 100
print('\nFault prevalence by turbine:')
print(turbine_summary.sort_values('fault_percentage', ascending=False))

# 5. Sensor redundancy.
print('\nRotor / generator speed correlation:')
print(data['rotor_speed_rpm'].corr(data['generator_speed_rpm']))

# 6. Wind-speed distribution.
wind = data['wind_speed_mps']
print('\nWind-speed summary:')
print(wind.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]))
print('Wind-speed skewness:', wind.skew())

# Verify the means reported incorrectly in notebook comments.
print('\nVerified sensor means:')
print(data[[
    'drivetrain_vibration_rms_mmps',
    'tower_vibration_mmps',
    'oil_particle_count',
]].mean())
```

## Overall interpretation

The EDA supports investigating multiple condition-monitoring signals, especially drivetrain vibration and oil particles, to detect rare fault conditions. Operating conditions and differences between turbines matter when interpreting measurements. This analysis describes associations with current fault labels; it does not establish advance warning of future failures or validated predictive performance.
