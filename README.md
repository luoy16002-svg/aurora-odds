# Aurora Odds

**Should you go outside and look up right now?** Aurora Odds answers that for your exact sky: the chance the
aurora is bright enough for your eyes or your phone camera in the next hour, whether it is dark and clear enough,
and which way to face and how high to look.

Live page: https://luoy16002-svg.github.io/aurora-odds/

The forecast is made by [TabPFN](https://github.com/PriorLabs/TabPFN), an open-weights tabular foundation model,
reading the solar wind measured at the L1 point 1.5 million km upstream. It is not trained on this problem: it
reads 2,000 labelled half-hours from 1998 to 2019 *in context* (8,000 in the evaluation below) and returns a full probability distribution for the
coming hour's geomagnetic activity. One distribution answers every latitude.

## How it works

1. **Upstream measurements.** NOAA's real-time solar wind feed (magnetic field and plasma from the active L1
   spacecraft, one-minute cadence) is averaged to five minutes and shifted to the time each parcel reaches Earth's
   bow shock, using the spacecraft's distance and the measured wind speed. Wind measured upstream reaches Earth
   20 to 80 minutes later, so part of the coming hour is already known.
2. **Features as they would be known.** `pipeline/features.py` summarises the field and plasma in four windows on
   the arrival-time axis: what is *already measured but not yet arrived*, the last 30 minutes, 30 to 90 minutes
   ago and 90 to 180 minutes ago. It uses the southward field (Bz), the Newell coupling function, speed, density
   and pressure, plus season and time of day. The same code builds the 1998-2026 history from NASA's OMNI data,
   using only samples that had been measured at L1 by each moment, so training and live inputs match.
3. **The target.** The highest [Hp30](https://kp.gfz.de/en/hp30-hp60) in the coming hour. Hp30 is GFZ Potsdam's
   half-hourly, open-ended version of the Kp index.
4. **TabPFN in context.** `pipeline/model.py` gives TabPFN v2 a stratified context of half-hours (storms are
   rare, so they are over-represented), then corrects the predicted distribution back to the true frequency of
   storms (a label-shift correction). `P(Hp30 ≥ k)` comes straight from the predicted distribution, for any `k`.
5. **Your sky.** The page converts your position to geomagnetic latitude, works out the Hp30 needed for the
   aurora to reach your horizon, your eyes or your zenith, and combines that with the sun's altitude and the
   current cloud cover from Open-Meteo. A GitHub Actions job reruns the model every ten minutes.

## Results

Test period: 2023-01-01 to 2026-09-18, 61,420 half-hours that none of the models saw (the solar maximum, with
the May and October 2024 storms). Brier skill score against climatology (higher is better, 0 = no skill), ROC AUC in brackets.
Events in the test period (half-hours whose next hour reached the level): ≥4: 9345 · ≥5: 3458 · ≥6: 1201 · ≥7: 485 · ≥8: 166.

| Model | Hp30 ≥ 4 | Hp30 ≥ 5 | Hp30 ≥ 6 | Hp30 ≥ 7 | Hp30 ≥ 8 |
|---|---:|---:|---:|---:|---:|
| TabPFN, 2,000-row context (live page) | 0.536 (0.948) | 0.517 (0.973) | 0.506 (0.986) | 0.559 (0.992) | 0.503 (0.992) |
| TabPFN, 8,000-row context | 0.555 (0.951) | 0.536 (0.975) | 0.519 (0.987) | 0.579 (0.993) | 0.529 (0.990) |
| LightGBM, trained on 370,431 rows | 0.556 (0.951) | 0.540 (0.974) | 0.506 (0.987) | 0.462 (0.989) | 0.338 (0.982) |
| Persistence of the last half-hour (oracle) | 0.344 (0.790) | 0.333 (0.777) | 0.337 (0.776) | 0.448 (0.811) | 0.439 (0.804) |

- TabPFN matches a LightGBM model trained on 46 to 185 times more rows for moderate activity and is clearly better for the
  rare strong storms (Hp30 ≥ 7 and ≥ 8), which are the nights that matter for anyone south of the auroral zone.
- The live page uses the 2,000-row context so a free 4-vCPU runner finishes in under a minute; it gives up about
  0.02 of skill against the 8,000-row context.
- Figures: `out/fig_skill.png`, `out/fig_reliability.png`, `out/fig_storm_may2024.png`, `out/fig_storm_oct2024.png`.


## Run it yourself

```bash
pip install -r requirements.txt
python pipeline/fetch_data.py      # NASA OMNI 5-min solar wind 1998-now, GFZ Hp30
python pipeline/build_table.py     # half-hourly features and targets
python pipeline/evaluate.py        # live config: 2,000-row context, 4 estimators; test on 2023-2026, baselines
python pipeline/evaluate.py 8000 8 uniform,stratified   # the larger evaluation config
python pipeline/replay.py          # storm replays for the page
python pipeline/figures.py         # charts in out/
python pipeline/nowcast.py         # live nowcast -> site/data/nowcast.json
```

Then serve `site/` with any static server.

## Credits and licences

- Code: MIT (see `LICENSE`).
- Built with TabPFN v2 by Prior Labs (model weights under the Prior Labs License).
- NASA/GSFC OMNI data via SPDF; GFZ Potsdam Hp30 index (CC BY 4.0, Yamazaki et al. 2022); NOAA SWPC real-time
  solar wind; Open-Meteo cloud cover.

This is a nowcast for curious people who want to go outside, not an operational space-weather product.
