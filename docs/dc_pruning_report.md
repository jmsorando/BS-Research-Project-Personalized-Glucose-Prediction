# Dc Feature Pruning Report
Generated: 2026-04-10 17:26


## Step 1 -- Biological Clusters
Clusters defined: 8
  carb_fractions: ['CHO', 'STAR', 'TOTSUG', 'FREE_SUGAR', 'ADDED_SUGAR', 'GLUC', 'FRUCT', 'SUCR', 'MALT', 'LACT', 'GALACT', 'OLIGO']
  energy_proxies: ['KCALS', 'KJ']
  macronutrients: ['PROT', 'FAT', 'ALCO', 'WATER']
  fibre: ['AOACFIB', 'ENGFIB']
  fat_subtypes: ['SATFAC', 'MONOFACc', 'POLYFACc', 'TOTn3PFAC', 'TOTn6PFAC', 'FACTRANS']
  micronutrients: ['MG', 'ZN', 'MN', 'FE', 'SE', 'VITD', 'CAFF']
  food_groups: ['totalVeg', 'totalFruit']
  derived_ratios: ['starch_fraction', 'sugar_fraction', 'free_sugar_fraction', 'rapid_glucose_equiv', 'intrinsic_sugar', 'fat_cho_ratio', 'protein_cho_ratio', 'fibre_cho_ratio', 'fat_sugar_ratio', 'protein_sugar_ratio', 'glycaemic_brake', 'n6_n3_ratio']
All Dc features assigned to exactly one cluster

## Step 2 -- VIF Analysis by Cluster
Thresholds: keep < 10 | flag 10-49 | candidate >= 50 | near-duplicate >= 100


### carb_fractions
  [~] CHO                            VIF =     33.4  (moderate -> monitor)
  [~] TOTSUG                         VIF =     18.4  (moderate -> monitor)
  [~] STAR                           VIF =     16.5  (moderate -> monitor)
  [ok] FRUCT                          VIF =      9.3  (acceptable)
  [ok] GLUC                           VIF =      8.6  (acceptable)
  [ok] SUCR                           VIF =      8.1  (acceptable)
  [ok] FREE_SUGAR                     VIF =      6.1  (acceptable)
  [ok] ADDED_SUGAR                    VIF =      4.9  (acceptable)
  [ok] LACT                           VIF =      2.0  (acceptable)
  [ok] MALT                           VIF =      1.8  (acceptable)
  [ok] GALACT                         VIF =      1.3  (acceptable)
  [ok] OLIGO                          VIF =      1.2  (acceptable)

### energy_proxies
  [X] KCALS                          VIF =  13236.2  (near-duplicate -> DROP candidate)
  [X] KJ                             VIF =  13236.2  (near-duplicate -> DROP candidate)

### macronutrients
  [ok] PROT                           VIF =      4.0  (acceptable)
  [ok] FAT                            VIF =      3.6  (acceptable)
  [ok] WATER                          VIF =      1.9  (acceptable)
  [ok] ALCO                           VIF =      1.3  (acceptable)

### fibre
  [ok] AOACFIB                        VIF =      8.3  (acceptable)
  [ok] ENGFIB                         VIF =      8.3  (acceptable)

### fat_subtypes
  [~] POLYFACc                       VIF =     14.8  (moderate -> monitor)
  [ok] TOTn6PFAC                      VIF =      9.2  (acceptable)
  [ok] MONOFACc                       VIF =      3.5  (acceptable)
  [ok] SATFAC                         VIF =      2.6  (acceptable)
  [ok] TOTn3PFAC                      VIF =      2.2  (acceptable)
  [ok] FACTRANS                       VIF =      2.0  (acceptable)

### micronutrients
  [ok] MG                             VIF =      6.3  (acceptable)
  [ok] MN                             VIF =      3.7  (acceptable)
  [ok] FE                             VIF =      3.0  (acceptable)
  [ok] ZN                             VIF =      2.7  (acceptable)
  [ok] SE                             VIF =      2.4  (acceptable)
  [ok] VITD                           VIF =      1.5  (acceptable)
  [ok] CAFF                           VIF =      1.1  (acceptable)

### food_groups
  [ok] totalVeg                       VIF =      1.0  (acceptable)
  [ok] totalFruit                     VIF =      1.0  (acceptable)

### derived_ratios
  [X] glycaemic_brake                VIF = 987968078.0  (near-duplicate -> DROP candidate)
  [X] protein_cho_ratio              VIF = 501620959.7  (near-duplicate -> DROP candidate)
  [X] fat_cho_ratio                  VIF = 75555080.4  (near-duplicate -> DROP candidate)
  [X] fibre_cho_ratio                VIF = 6688540.2  (near-duplicate -> DROP candidate)
  [ok] protein_sugar_ratio            VIF =      6.3  (acceptable)
  [ok] fat_sugar_ratio                VIF =      6.0  (acceptable)
  [ok] sugar_fraction                 VIF =      4.8  (acceptable)
  [ok] intrinsic_sugar                VIF =      4.0  (acceptable)
  [ok] free_sugar_fraction            VIF =      4.0  (acceptable)
  [ok] rapid_glucose_equiv            VIF =      3.2  (acceptable)
  [ok] starch_fraction                VIF =      1.9  (acceptable)
  [ok] n6_n3_ratio                    VIF =      1.0  (acceptable)

VIF table saved -> output\results\pruning\vif_by_cluster.csv

Summary: 6 near-duplicates (VIF>=100), 6 total candidates (VIF>=50)

## Step 3 -- SHAP for Dc-only model

Dc-only SHAP ranking (top 20):
  # 1  glycaemic_brake                 mean|SHAP| = 6.1162
  # 2  protein_cho_ratio               mean|SHAP| = 4.7678
  # 3  sugar_fraction                  mean|SHAP| = 4.2809
  # 4  fat_sugar_ratio                 mean|SHAP| = 3.6543
  # 5  ZN                              mean|SHAP| = 3.5222
  # 6  SE                              mean|SHAP| = 3.3023
  # 7  fat_cho_ratio                   mean|SHAP| = 3.2177
  # 8  FE                              mean|SHAP| = 3.2112
  # 9  SUCR                            mean|SHAP| = 3.1747
  #10  MG                              mean|SHAP| = 3.1482
  #11  PROT                            mean|SHAP| = 3.0584
  #12  POLYFACc                        mean|SHAP| = 3.0545
  #13  AOACFIB                         mean|SHAP| = 3.0481
  #14  WATER                           mean|SHAP| = 3.0168
  #15  intrinsic_sugar                 mean|SHAP| = 2.8802
  #16  MONOFACc                        mean|SHAP| = 2.8037
  #17  fibre_cho_ratio                 mean|SHAP| = 2.4787
  #18  MN                              mean|SHAP| = 2.3764
  #19  ENGFIB                          mean|SHAP| = 2.3141
  #20  CHO                             mean|SHAP| = 2.2606

SHAP ranking saved -> output\results\pruning\dc_shap_ranking.csv
Plot saved -> output\plots\pruning\dc_only_shap_beeswarm.png

## Step 4 -- Combined VIF + SHAP Decision Table

Feature                        Cluster                   VIF     SHAP  G.Rank Decision  Reason
------------------------------------------------------------------------------------------------------------------------
[KEEP] SUCR                         carb_fractions            8.2   3.1747 #   9   KEEP      VIF=8.2 < 10 -- acceptable collinearity
[KEEP] CHO                          carb_fractions           33.4   2.2606 #  20   KEEP      VIF=33.4 -- moderate, acceptable
[KEEP] STAR                         carb_fractions           16.5   2.2525 #  22   KEEP      VIF=16.5 -- moderate, acceptable
[KEEP] MALT                         carb_fractions            1.8   2.1556 #  23   KEEP      VIF=1.8 < 10 -- acceptable collinearity
[KEEP] TOTSUG                       carb_fractions           18.4   1.8984 #  27   KEEP      VIF=18.4 -- moderate, acceptable
[KEEP] GLUC                         carb_fractions            8.6   1.8610 #  30   KEEP      VIF=8.6 < 10 -- acceptable collinearity
[KEEP] FRUCT                        carb_fractions            9.2   1.8529 #  31   KEEP      VIF=9.2 < 10 -- acceptable collinearity
[KEEP] LACT                         carb_fractions            2.0   1.6356 #  34   KEEP      VIF=2.0 < 10 -- acceptable collinearity
[KEEP] GALACT                       carb_fractions            1.3   1.3343 #  38   KEEP      VIF=1.3 < 10 -- acceptable collinearity
[KEEP] OLIGO                        carb_fractions            1.1   1.1956 #  40   KEEP      VIF=1.1 < 10 -- acceptable collinearity
[KEEP] ADDED_SUGAR                  carb_fractions            4.9   1.1255 #  41   KEEP      VIF=4.9 < 10 -- acceptable collinearity
[KEEP] FREE_SUGAR                   carb_fractions            6.1   1.0648 #  42   KEEP      VIF=6.1 < 10 -- acceptable collinearity
[KEEP] glycaemic_brake              derived_ratios       987968078.0   6.1162 #   1   KEEP      VIF=987968078.0 but global SHAP rank=#1 -- too important to drop
[KEEP] protein_cho_ratio            derived_ratios       501620959.8   4.7678 #   2   KEEP      VIF=501620959.8 but global SHAP rank=#2 -- too important to drop
[KEEP] sugar_fraction               derived_ratios            4.8   4.2809 #   3   KEEP      VIF=4.8 < 10 -- acceptable collinearity
[KEEP] fat_sugar_ratio              derived_ratios            6.0   3.6543 #   4   KEEP      VIF=6.0 < 10 -- acceptable collinearity
[DROP] fat_cho_ratio                derived_ratios       75555080.4   3.2177 #   7   DROP      VIF=75555080.4 >= 100 and not top-SHAP in cluster (cluster rank #5/12)
[KEEP] intrinsic_sugar              derived_ratios            4.0   2.8802 #  15   KEEP      VIF=4.0 < 10 -- acceptable collinearity
[DROP] fibre_cho_ratio              derived_ratios       6688540.2   2.4787 #  17   DROP      VIF=6688540.2 >= 100 and not top-SHAP in cluster (cluster rank #7/12)
[KEEP] rapid_glucose_equiv          derived_ratios            3.2   2.2547 #  21   KEEP      VIF=3.2 < 10 -- acceptable collinearity
[KEEP] starch_fraction              derived_ratios            1.9   2.0056 #  24   KEEP      VIF=1.9 < 10 -- acceptable collinearity
[KEEP] protein_sugar_ratio          derived_ratios            6.3   1.8253 #  32   KEEP      VIF=6.3 < 10 -- acceptable collinearity
[KEEP] n6_n3_ratio                  derived_ratios            1.1   1.4870 #  36   KEEP      VIF=1.1 < 10 -- acceptable collinearity
[KEEP] free_sugar_fraction          derived_ratios            4.0   0.8084 #  46   KEEP      VIF=4.0 < 10 -- acceptable collinearity
[KEEP] KCALS                        energy_proxies        13236.2   1.8785 #  29   KEEP      VIF=13236.2 >= 50 but top half of cluster by SHAP (rank #1/2) -- keep, note in paper
[DROP] KJ                           energy_proxies        13236.2   1.2853 #  39   DROP      VIF=13236.2 >= 100 and not top-SHAP in cluster (cluster rank #2/2)
[KEEP] POLYFACc                     fat_subtypes             14.8   3.0545 #  12   KEEP      VIF=14.8 -- moderate, acceptable
[KEEP] MONOFACc                     fat_subtypes              3.5   2.8037 #  16   KEEP      VIF=3.5 < 10 -- acceptable collinearity
[KEEP] SATFAC                       fat_subtypes              2.6   1.9789 #  25   KEEP      VIF=2.6 < 10 -- acceptable collinearity
[KEEP] FACTRANS                     fat_subtypes              2.0   1.8941 #  28   KEEP      VIF=2.0 < 10 -- acceptable collinearity
[KEEP] TOTn3PFAC                    fat_subtypes              2.2   1.7376 #  33   KEEP      VIF=2.2 < 10 -- acceptable collinearity
[KEEP] TOTn6PFAC                    fat_subtypes              9.2   1.6191 #  35   KEEP      VIF=9.2 < 10 -- acceptable collinearity
[KEEP] AOACFIB                      fibre                     8.3   3.0481 #  13   KEEP      VIF=8.3 < 10 -- acceptable collinearity
[KEEP] ENGFIB                       fibre                     8.3   2.3141 #  19   KEEP      VIF=8.3 < 10 -- acceptable collinearity
[KEEP] totalFruit                   food_groups               1.0   0.9586 #  43   KEEP      VIF=1.0 < 10 -- acceptable collinearity
[KEEP] totalVeg                     food_groups               1.0   0.9507 #  44   KEEP      VIF=1.0 < 10 -- acceptable collinearity
[KEEP] PROT                         macronutrients            4.0   3.0584 #  11   KEEP      VIF=4.0 < 10 -- acceptable collinearity
[KEEP] WATER                        macronutrients            1.9   3.0168 #  14   KEEP      VIF=1.9 < 10 -- acceptable collinearity
[KEEP] FAT                          macronutrients            3.6   1.9086 #  26   KEEP      VIF=3.6 < 10 -- acceptable collinearity
[KEEP] ALCO                         macronutrients            1.4   0.0711 #  47   KEEP      VIF=1.4 < 10 -- acceptable collinearity
[KEEP] ZN                           micronutrients            2.7   3.5222 #   5   KEEP      VIF=2.7 < 10 -- acceptable collinearity
[KEEP] SE                           micronutrients            2.4   3.3023 #   6   KEEP      VIF=2.4 < 10 -- acceptable collinearity
[KEEP] FE                           micronutrients            3.0   3.2112 #   8   KEEP      VIF=3.0 < 10 -- acceptable collinearity
[KEEP] MG                           micronutrients            6.3   3.1482 #  10   KEEP      VIF=6.3 < 10 -- acceptable collinearity
[KEEP] MN                           micronutrients            3.7   2.3764 #  18   KEEP      VIF=3.7 < 10 -- acceptable collinearity
[KEEP] VITD                         micronutrients            1.5   1.3361 #  37   KEEP      VIF=1.5 < 10 -- acceptable collinearity
[KEEP] CAFF                         micronutrients            1.1   0.8225 #  45   KEEP      VIF=1.1 < 10 -- acceptable collinearity

Result: KEEP 44 features, DROP 3 features
Dropped: ['KJ', 'fat_cho_ratio', 'fibre_cho_ratio']

## Step 5 -- VIF vs SHAP scatter plot
Plot saved -> output\plots\pruning\vif_vs_shap_scatter.png (.svg also)

## Step 6 -- CV Performance: Full vs Pruned

Feature counts:
  Full model:         88 features
  Pruned Dc + G/Dt/P: 85 features (44 Dc + 41 other)
  Pruned Dc only:     44 features

  Full model (all features)                      R2=0.084+/-0.037  MAE=53.0+/-2.0
  Pruned Dc + G + Dt + P + Interactions          R2=0.079+/-0.038  MAE=53.1+/-2.0
  Pruned Dc only (no G)                          R2=-0.157+/-0.051  MAE=60.8+/-1.8

dR2  (full vs pruned): 0.0056
dMAE (full vs pruned): 0.15 mmol*min
MARGINAL -- 0.005 <= dR2 < 0.01 -- acceptable but note in paper

CV comparison saved -> output\results\pruning\cv_comparison_full_vs_pruned.csv

## Step 7 -- Update config.py

Pruned DC_RAW    (34 features):    ['CHO', 'STAR', 'TOTSUG', 'FREE_SUGAR', 'ADDED_SUGAR', 'GLUC', 'FRUCT', 'SUCR', 'MALT', 'LACT', 'GALACT', 'OLIGO', 'PROT', 'FAT', 'KCALS', 'ALCO', 'WATER', 'AOACFIB', 'ENGFIB', 'SATFAC', 'MONOFACc', 'POLYFACc', 'TOTn3PFAC', 'TOTn6PFAC', 'FACTRANS', 'MG', 'ZN', 'MN', 'FE', 'SE', 'VITD', 'CAFF', 'totalVeg', 'totalFruit']
Pruned DC_RATIOS (10 features): ['starch_fraction', 'sugar_fraction', 'free_sugar_fraction', 'rapid_glucose_equiv', 'intrinsic_sugar', 'protein_cho_ratio', 'fat_sugar_ratio', 'protein_sugar_ratio', 'glycaemic_brake', 'n6_n3_ratio']
Original config backed up -> config_pre_pruning_backup.py
config.py updated with pruned DC_RAW and DC_RATIOS
   DC_RAW:    35 -> 34 features
   DC_RATIOS: 12 -> 10 features

## Step 8 -- Methods Section Text

Methods text saved -> output\results\pruning\methods_section_text.txt

## Final Summary
============================================================
Original Dc features:  47
Features kept:         44
Features dropped:      3
Reduction:             6%
dR2 (full vs pruned):  0.0056  MARGINAL
config.py updated:     yes

Outputs:
  output\results\pruning\vif_by_cluster.csv
  output\results\pruning\dc_shap_ranking.csv
  output\results\pruning\pruning_decisions.csv
  output\results\pruning\cv_comparison_full_vs_pruned.csv
  output\results\pruning\methods_section_text.txt
  output\plots\pruning\dc_only_shap_beeswarm.png
  output\plots\pruning\vif_vs_shap_scatter.png (.svg also)
  output/results/dc_pruning_report.md
============================================================
