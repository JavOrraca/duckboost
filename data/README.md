# Example datasets (offline tests)

Thin CSV copies for **offline SQL tests** under `test/sql/duckboost/`. Getting Started and the website vignettes load the same data from public HTTPS URLs instead — see the [example datasets vignette](https://javorraca.github.io/duckboost/vignettes/datasets.html).

| File | Rows | Source | License |
| --- | --- | --- | --- |
| `iris.csv` | 150 | R `datasets::iris` (Anderson, 1935; Fisher, 1936), via [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/) | Public domain |
| `penguins.csv` | 344 | [palmerpenguins](https://github.com/allisonhorst/palmerpenguins) `inst/extdata/penguins.csv` (Horst, Hill, and Gorman, 2020) | CC0 1.0 |

`iris.csv` renames R's columns to snake_case (`Sepal.Length` becomes `sepal_length`) and drops the row-name column. Values are R's, which differ from the UCI Machine Learning Repository copy in rows 35 and 38.

`penguins.csv` is unchanged from the package and writes missing values as `NA`. Tests read it with `read_csv('data/penguins.csv', nullstr = 'NA')`.

Please cite the palmerpenguins package and the original study when you use the penguins data:

- Horst, A. M., Hill, A. P., and Gorman, K. B. (2020). palmerpenguins: Palmer Archipelago (Antarctica) penguin data. R package version 0.1.0. doi:10.5281/zenodo.3960218.
- Gorman, K. B., Williams, T. D., and Fraser, W. R. (2014). Ecological sexual dimorphism and environmental variability within a community of Antarctic penguins (genus *Pygoscelis*). *PLoS ONE*, 9(3), e90081. doi:10.1371/journal.pone.0090081.
