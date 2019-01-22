list.of.cran.packages <- c("tibble")
new.cran.packages <- list.of.cran.packages[!(list.of.cran.packages %in% installed.packages()[,"Package"])]
if(length(new.cran.packages))
    install.packages(new.cran.packages, repos="http://cran.rstudio.com", dependencies = TRUE)

list.of.bioc.packages <- c("Biobase", "GEOquery", "rhdf5", "limma", "edgeR")
new.bioc.packages <- list.of.bioc.packages[!(list.of.bioc.packages %in% installed.packages()[,"Package"])]
if(length(new.bioc.packages)){
  source("https://bioconductor.org/biocLite.R")
  biocLite()
  biocLite(pkgs = new.bioc.packages, ask = FALSE)
}