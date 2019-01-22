# Version info: R 3.2.3, Biobase 2.30.0, GEOquery 2.40.0, limma 3.26.8
# R scripts generated  Thu Jun 28 07:40:34 EDT 2018

################################################################
#   Differential expression analysis with limma
list.of.bioc.packages <- c("Biobase", "GEOquery", "limma")
new.bioc.packages <- list.of.bioc.packages[!(list.of.bioc.packages %in% installed.packages()[,"Package"])]
if(length(new.bioc.packages)){
  source("https://bioconductor.org/biocLite.R")
  biocLite(pkgs = new.bioc.packages, ask = FALSE)
}

library(Biobase)
library(GEOquery)
library(limma)

args = commandArgs(trailingOnly=TRUE)
# test if there is at least one argument: if not, return an error
if (length(args) != 4) {
  stop("Usage: Rscript analyse_microarray.R <gse_id> <gpl_id> <grouping> <output>", call.=FALSE)
}

gse_id = args[1]
gpl_id = args[2]
grouping = args[3]
output = args[4]

# load series and platform data from GEO

gset <- getGEO(gse_id, GSEMatrix =TRUE, AnnotGPL=TRUE)
if (length(gset) > 1) idx <- grep(gpl_id, attr(gset, "names")) else idx <- 1
gset <- gset[[idx]]

# make proper column names to match toptable
fvarLabels(gset) <- make.names(fvarLabels(gset))

# group names for all samples
gsms <- grouping
sml <- c()
for (i in 1:nchar(gsms)) { sml[i] <- substr(gsms,i,i) }

# eliminate samples marked as "X"
sel <- which(sml != "X")
sml <- sml[sel]
gset <- gset[ ,sel]

# log2 transform
ex <- exprs(gset)
qx <- as.numeric(quantile(ex, c(0., 0.25, 0.5, 0.75, 0.99, 1.0), na.rm=T))
LogC <- (qx[5] > 100) ||
          (qx[6]-qx[1] > 50 && qx[2] > 0) ||
          (qx[2] > 0 && qx[2] < 1 && qx[4] > 1 && qx[4] < 2)
if (LogC) { ex[which(ex <= 0)] <- NaN
  exprs(gset) <- log2(ex) }

# set up the data and proceed with analysis
sml <- paste("G", sml, sep="")    # set group names
fl <- as.factor(sml)
gset$description <- fl
design <- model.matrix(~ description + 0, gset)
colnames(design) <- levels(fl)
fit <- lmFit(gset, design)
cont.matrix <- makeContrasts(G1-G0, levels=design)
fit2 <- contrasts.fit(fit, cont.matrix)
fit2 <- eBayes(fit2, 0.01)
tT <- topTable(fit2, adjust="fdr", sort.by="B", number=Inf)

filter.toptable <- function(tt){

  na.filter <- !is.na(tt$AveExpr)

  tt <- tt[na.filter,]

  ## take the top 2/3 of expressed probes.
  averages = tt$AveExpr

  a <- sort(averages)
  min.ave.exprs <- a[length(a)/3]
  exprs.filter <- averages>min.ave.exprs

  gSymbols <- tt$Gene.symbol
  no.symbol.filter <- !gSymbols == ""

  sags <- split(tt[,"AveExpr", drop=F], gSymbols)
  highest.probes.keep <- unlist(lapply(sags, function(X){return(row.names(X)[which.max(unlist(X))])}))

  highest.probe.filter <- row.names(tt) %in% highest.probes.keep

  combined.filter <- exprs.filter & no.symbol.filter & highest.probe.filter

  return(tt[combined.filter,])
}

#remove low expressed / redundant probes and genes without symbols
tTF <- filter.toptable(tT)

#readjust P value
tTF$adj.P.Val <- p.adjust(tTF$P.Value, method = "BH")

tT <- subset(tTF, select=c("adj.P.Val","P.Value","t","B","logFC","Gene.symbol"))
write.table(tT, file=sprintf(output), row.names=F, sep="\t")