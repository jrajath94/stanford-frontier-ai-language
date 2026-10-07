# Lab key , U10

Execution-verified outputs from `labs/u10_lab_run.py`, run 2026-10-07
(CPython, numpy 1.26.4, CPU). Re-run the script to confirm.

## T1 , Wilson intervals

78/100: [0.6893, 0.8500]. 780/1000: [0.7533, 0.8046]. Same point,
honest claim.

## T2 , contamination

Reported 0.75, clean-only 0.70. The 40 leaked items inflate the
score by 0.05.

## T3 , kappa

0.551. Chance alone agrees 0.51 of the time, so 0.78 observed is
moderate agreement.

## T4 , slices

Delta A: -0.26. Delta B: +0.08. The mean hides slice A.

## T5 , sample size

n = 2401 for margin 0.02 at 95 percent.

## T6 , Brier

0.17. The judge's confidence bins miscalibrate.
