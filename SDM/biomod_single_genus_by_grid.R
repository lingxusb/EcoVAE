# ==== Load Required Libraries ====
suppressPackageStartupMessages({
  library(biomod2)
  library(tidyverse)
  library(terra)
})

# ==== Set Working Directory and Output Directory ====
outdir <- "output/"
# if (!file.exists(outdir)) {
#   dir.create(outdir)
#   message("Output directory created: ", outdir)
# }

# ==== Parse Command Line Arguments ====
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Two arguments (genus and group) are required.")
} else {
  sp <- args[1]
  temp_g <- args[2]
}

# ==== Define Species Name ====
#sp <- gsub('^["\\\\]+|["\\\\]+$', '', args[1])
cat("Running for genus:", sp, "and group:", temp_g, "\n")
# For testing only:
#sp <- "Clappertonia"#"Verticordia.auriculata" 
#temp_g <- "0"
df_name <- file.path("data/1000genusocc", paste0(sp, "_occ.csv"))

# ==== Load Occurrence Data ====
df <- read.csv(df_name)

# ==== Select Environmental Variables ====
env_names <- paste0("bio", 1:19)
env_files <- file.path("data/env_data", paste0(env_names, ".tif"))

# ==== Split Training and Testing Data ====
# Here we split based on the designate input g number
df_training <- df %>% filter(cv != temp_g & presence == 1)
df_testing  <- df %>% filter(cv == temp_g)
sum_input <- data.frame(species=sp, num_training=sum(df_training$presence), num_testing=sum(df_testing$presence), cv_testing=temp_g)
write.csv(sum_input, file.path(outdir, paste0(sp, "_g", temp_g,"_sum_input.csv")), row.names = F)

# ==== Prepare Response Data ====
myRespName <- paste0(sp,"_g",temp_g)
myResp     <- as.numeric(df_training$presence)
myRespXY   <- df_training %>% select(x, y)

# ==== Prepare Evaluation Data ====
myEvaResp <- as.numeric(df_testing$presence)
myEvaResp[myEvaResp == 0] <- NA
replace_indices <- sample(which(is.na(myEvaResp)), 50)
myEvaResp[replace_indices] <- 0

myEvaRespXY <- df_testing[!is.na(myEvaResp), c("x", "y")]
myEvaExpl   <- df_testing[!is.na(myEvaResp), env_names]
myEvaResp   <- na.omit(myEvaResp)
cat("Evaluation sample size:", length(myEvaResp), "\n")

# ==== Compute Pairwise Angular Distances ====
angular_distance_deg <- function(lat1, lon1, lat2, lon2) {
  rad <- pi / 180
  lat1 <- lat1 * rad; lon1 <- lon1 * rad
  lat2 <- lat2 * rad; lon2 <- lon2 * rad
  
  dlat <- lat2 - lat1
  dlon <- lon2 - lon1
  
  a <- sin(dlat / 2)^2 + cos(lat1) * cos(lat2) * sin(dlon / 2)^2
  c <- 2 * atan2(sqrt(a), sqrt(1 - a))
  return(c * (180 / pi))
}

n <- nrow(myRespXY)
dists <- matrix(0, n, n)

for (i in 1:(n - 1)) {
  for (j in (i + 1):n) {
    dists[i, j] <- angular_distance_deg(myRespXY$y[i], myRespXY$x[i], myRespXY$y[j], myRespXY$x[j])
  }
}

dists_v <- dists[dists != 0]
max_deg <- max(dists)
md_deg  <- median(dists_v)

write.table(data.frame(max = max_deg, median = md_deg),
            file = file.path(outdir, paste0(myRespName, "_presence_dist.txt")),
            row.names = FALSE)

message("Max distance: ", max_deg, "°; Median distance: ", md_deg, "°")

# ==== Load and Crop Environmental Raster Stack ====
exp_deg <- 10#3 * max_deg
# extent_train <- extent(
#   min(myRespXY$x) - exp_deg, max(myRespXY$x) + exp_deg,
#   min(myRespXY$y) - exp_deg, max(myRespXY$y) + exp_deg
# )

# using terra package, but not raster package
extent_train <- terra::ext(
  min(myRespXY$x) - exp_deg, max(myRespXY$x) + exp_deg,
  min(myRespXY$y) - exp_deg, max(myRespXY$y) + exp_deg
)
bioclim_stack <- terra::rast(env_files)
bioclim_stack <- terra::crop(bioclim_stack, extent_train)
names(bioclim_stack) <- env_names


# ==== Select Environmental Variables if needed ====
num_occ <- nrow(myRespXY)
# calculate the correlation between variables and select 5-10 variables as input
if(num_occ<200){
  target_n = floor(num_occ/10)
  bioclim_mat <- as.data.frame(bioclim_stack, na.rm = TRUE)
  
  cor_env <- cor(bioclim_mat,method = "spearman")
  cor_env_df <- as.data.frame(as.table(cor_env)) %>%
    filter(Var1 != Var2) %>%        # Remove self-correlations
    #filter(abs(Freq) > 0.8) %>%     # Filter by correlation threshold
    rowwise() %>%
    mutate(pair = paste(sort(c(Var1, Var2)), collapse = "_")) %>%
    ungroup() %>%
    distinct(pair, .keep_all = TRUE) %>%  # Remove duplicate pairs
    select(Var1, Var2, correlation = Freq)
  write.csv(cor_env_df, file = file.path(outdir, paste0(myRespName, "_cor_env_df.csv")), row.names = F)
  # write.csv(var_frequency, file = "var_frequency.csv", row.names = F)
  
  # Define correlation threshold
  threshold <- 0.7
  
  # Ensure symmetric correlation pairs by adding flipped versions
  cor_df_flipped <- cor_env_df
  names(cor_df_flipped)[1:2] <- c("Var2", "Var1")
  
  cor_all <- bind_rows(cor_env_df, cor_df_flipped)
  cor_all$Var1 <- as.character(cor_all$Var1)
  cor_all$Var2 <- as.character(cor_all$Var2)
  
  # Filter only highly correlated pairs
  high_cor_pairs <- cor_all %>%
    filter(abs(correlation) > threshold)
  
  # Get a unique list of all variable names
  all_vars <- as.character(unique(c(cor_all$Var1, cor_all$Var2)))
  
  # Count how many highly correlated partners each variable has
  cor_counts <- high_cor_pairs %>%
    group_by(Var1) %>%
    summarise(HighCorrCount = n()) %>%
    arrange(desc(HighCorrCount))
  
  # Include variables with 0 high correlations
  cor_counts <- tibble(Var1 = all_vars) %>%
    left_join(cor_counts, by = "Var1") %>%
    mutate(HighCorrCount = ifelse(is.na(HighCorrCount), 0, HighCorrCount)) %>%
    arrange(desc(HighCorrCount))
  
  # Initialize selection process
  selected_vars <- c()
  excluded_vars <- c()
  remaining_vars <- as.character(cor_counts$Var1)
  
  # Iterative selection loop
  while (length(remaining_vars) > 0) {
    current_var <- remaining_vars[1]
    selected_vars <- c(selected_vars, current_var)
    
    # Find variables highly correlated with current_var
    correlated_vars <- high_cor_pairs %>%
      filter(Var1 == current_var) %>%
      pull(Var2)
    
    # Exclude current_var and its correlated variables
    excluded_vars <- unique(c(excluded_vars, current_var, correlated_vars))
    
    # Update remaining variables
    remaining_vars <- setdiff(remaining_vars, excluded_vars)
  }
  if (length(selected_vars) > target_n){
    selected_vars <- selected_vars[1:target_n]
  }
  # Final output
  cat("Selected variables:\n")
  print(selected_vars)
  write.csv(data.frame(species=sp, n_env=length(selected_vars)), file.path(outdir, paste0(myRespName, "_n_env.csv")), row.names = F)
  myExpl <- subset(bioclim_stack, selected_vars)
}else{
  myExpl <- bioclim_stack
}


# ==== Format Data for biomod2 ====
myBiomodData.multi <- BIOMOD_FormatingData(
  resp.var = myResp,
  expl.var = myExpl,
  resp.xy = myRespXY,
  resp.name = myRespName,
  PA.nb.rep = 1,
  PA.nb.absences = 5000,
  PA.strategy = 'random',
  eval.resp.var = myEvaResp,
  eval.resp.xy = myEvaRespXY,
  eval.expl.var = myEvaExpl
)

# ==== Plot Input Data ====
pdf(file = file.path(outdir, paste0(myRespName, "_input.pdf")), width = 8, height = 6)
plot(myBiomodData.multi)
dev.off()

# ==== Run Models ====
start_time <- Sys.time()

myBiomodModelOut <- BIOMOD_Modeling(
  myBiomodData.multi,
  models = c('GLM', 'MAXENT', 'RFd'),
  CV.strategy = "block",
  CV.do.full.models = T,
  OPT.strategy = "bigboss",
  metric.eval = c('TSS', 'ROC', 'BOYCE'),
  var.import = 1,
  modeling.id = paste0(myRespName, "_Modeling"),
  seed.val = 123,
  nb.cpu = 8
)

myBiomodModelOut2 <- BIOMOD_EnsembleModeling(
  myBiomodModelOut,
  models.chosen = "all",
  em.by = "all",
  em.algo = c("EMwmean"),
  metric.select = "ROC",
  metric.select.thresh = c(0.7),
  metric.select.table = NULL,
  metric.select.dataset = "calibration",
  metric.eval = c("TSS", "ROC","BOYCE"),
  var.import = 1,
  EMci.alpha = 0.05,
  EMwmean.decay = "proportional",
  nb.cpu = 8,
  seed.val = 123,
  do.progress = TRUE
)


# ==== Save Evaluation and Variable Importance ====
eva <- get_evaluations(myBiomodModelOut)
eva2 <- get_evaluations(myBiomodModelOut2)
var_importance <- get_variables_importance(myBiomodModelOut)
var_importance2 <- get_variables_importance(myBiomodModelOut2)

write.csv(eva, file.path(outdir, paste0(myRespName, "_model_evaluation.csv")), row.names = FALSE)
write.csv(var_importance, file.path(outdir, paste0(myRespName, "_var_importance.csv")), row.names = FALSE)
write.csv(eva2, file.path(outdir, paste0(myRespName, "_model_evaluation2.csv")), row.names = FALSE)
write.csv(var_importance2, file.path(outdir, paste0(myRespName, "_var_importance2.csv")), row.names = FALSE)

# ==== Runtime Summary ====
end_time <- Sys.time()
runtime <- end_time - start_time

runtime_table <- data.frame(
  Taxa = myRespName,
  Start_Time = start_time,
  End_Time = end_time,
  Duration = runtime
)

write.csv(runtime_table,
          file = file.path(outdir, paste0(myRespName, "_proc_time.csv")),
          row.names = FALSE)
