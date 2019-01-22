# R script to download selected samples
# Copy code and run on a local machine to initiate download
# Check for dependencies and install if missing
list.of.cran.packages <- c("tibble")
new.cran.packages <- list.of.cran.packages[!(list.of.cran.packages %in% installed.packages()[,"Package"])]
if(length(new.cran.packages))
    install.packages(new.cran.packages, dependencies = TRUE)

list.of.bioc.packages <- c("rhdf5", "limma", "edgeR")
new.bioc.packages <- list.of.bioc.packages[!(list.of.bioc.packages %in% installed.packages()[,"Package"])]
if(length(new.bioc.packages)){
  source("https://bioconductor.org/biocLite.R")
  biocLite(pkgs = new.bioc.packages, ask = FALSE)
}

packages <- c("rhdf5", "limma", "edgeR")
if (length(setdiff(packages, rownames(installed.packages()))) > 0) {
    source("https://bioconductor.org/biocLite.R")
    biocLite("rhdf5")
}
library("rhdf5")
library("limma")
library("edgeR")
library("tibble")

args = commandArgs(trailingOnly=TRUE)
# test if there is at least one argument: if not, return an error
if (length(args) != 3) {
  stop("Usage: Rscript analyserna_seq.R <gsm_ids> <grouping> <output>", call.=FALSE)
}

gsm_ids = unlist(strsplit(args[1], ","))
grouping = unlist(strsplit(args[2], ","))
output = args[3]

destination_file = "human_matrix.h5"

# Retrieve information from compressed data
samples = h5read(destination_file, "meta/Sample_geo_accession")
# Identify columns to be extracted
sample_locations = which(samples %in% gsm_ids)
genes = h5read(destination_file, "meta/genes")

# extract gene expression from compressed data
expression = h5read(destination_file, "data/expression", index=list(1:length(genes), sample_locations))
H5close()
rownames(expression) = genes
colnames(expression) = samples[sample_locations]

dge <- DGEList(counts=expression)
design <- model.matrix(~ 0+factor(c(grouping)))
colnames(design) <- c("ctrl", "pert")

# Normalisation and Filtering
keep <- filterByExpr(dge, design)
dge <- dge[keep,,keep.lib.sizes=FALSE]
dge <- calcNormFactors(dge)

logCPM <- cpm(dge, log=TRUE, prior.count=3)
fit <- lmFit(logCPM, design)
fit <- eBayes(fit, trend=TRUE)
tT <- topTable(fit, coef=ncol(design), adjust="fdr", sort.by="B", number=250)
tT = rownames_to_column(tT, "Gene")

# tT <- subset(tT, select=c("ID","adj.P.Val","P.Value","t","B","logFC","Gene.symbol","Gene.title"))
write.table(tT, file=sprintf(output), row.names=F, sep="\t")