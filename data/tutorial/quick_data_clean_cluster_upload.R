# GBIF data filtering ####

# Load the packages
library("tidyverse")
library("CoordinateCleaner")
library("rgbif")

args=(commandArgs(TRUE)) # dbName
if(length(args)==0){
  print("No arguments supplied.")
}else{
  for(i in 1:length(args)){
    eval(parse(text=args[[i]]))
  }
}
# get species name and coordinate
setwd("data/")
dbName <- "0013312-250415084134356"

# Load the data
db <- occ_download_import(as.download(paste0(dbName,".zip")))

sp_occ <- db %>% dplyr::select(c("kingdom", "phylum", "class", "order", "family", "genus", "species", "infraspecificEpithet","taxonRank", "scientificName","taxonKey", "speciesKey","countryCode", "decimalLatitude", "decimalLongitude", "year","basisOfRecord","establishmentMeans","issue")) %>% 
  filter(is.na(decimalLatitude)==F, 
         basisOfRecord %in% c("HUMAN_OBSERVATION", "PRESERVED_SPECIMEN"),
         !establishmentMeans %in% c("introduced","uncertain","vagrant"))%>%
   distinct()
   
write.csv(sp_occ, paste0(dbName,"_simplified.csv"),row.names = F)
sp_occ <- read.csv(paste0(dbName,"_simplified.csv"))

# Cleaning suspicious records using Coordinatecleaner package
sp_occ_flagged <- clean_coordinates(
  sp_occ,
  lon = "decimalLongitude",
  lat = "decimalLatitude",
  species = "species",
  countries = NULL,
  tests = c("capitals", "centroids", "equal", "gbif", "institutions", #"outliers",
            "zeros"), #"seas",
  verbose = TRUE)

sp_occ_clean <- sp_occ_flagged %>% 
  filter(taxonRank=="SPECIES", .summary==T)

sp_occ_clean <- sp_occ_clean[,1:18] %>% 
  mutate(presence=rep(1, length=nrow(sp_occ_clean)))

sp_occ_clean_observation <- sp_occ_clean %>% filter(basisOfRecord == "HUMAN_OBSERVATION")
sp_occ_clean_specimen <- sp_occ_clean %>% filter(basisOfRecord == "PRESERVED_SPECIMEN")

write.csv(sp_occ_clean_observation, paste0(dbName,"_clean_observation.csv"),row.names = F)
write.csv(sp_occ_clean_specimen, paste0(dbName,"_clean_specimen.csv"),row.names = F)

# Generate summary report
sink(paste0(dbName, "_summary.txt"))
cat(paste0("Observation - number of records: ", nrow(sp_occ_clean_observation), "; number of species: ", length(unique(sp_occ_clean_observation$speciesKey))))
cat("\n")
cat(paste0(max(sp_occ_clean_observation$decimalLatitude), ",",min(sp_occ_clean_observation$decimalLatitude)))
cat("\n")
cat(paste0(max(sp_occ_clean_observation$decimalLongitude), ",", min(sp_occ_clean_observation$decimalLongitude)))
cat("\n")
cat(paste0("Specimen - number of records: ", nrow(sp_occ_clean_specimen), "; number of species: ", length(unique(sp_occ_clean_specimen$speciesKey))))
cat("\n")
cat(paste0(max(sp_occ_clean_specimen$decimalLatitude), ",", min(sp_occ_clean_specimen$decimalLatitude)))
cat("\n")
cat(paste0(max(sp_occ_clean_specimen$decimalLongitude), ",", min(sp_occ_clean_specimen$decimalLongitude)))
sink()


# Generate taxon list ####
sp_list <- sp_occ_clean %>% select(c("kingdom", "phylum", "class", "order", "family", "genus", "species", "infraspecificEpithet","taxonRank", "scientificName","taxonKey")) %>% distinct()
write.csv(sp_list, paste0(dbName,"_clean_splist.csv"),row.names = F)
