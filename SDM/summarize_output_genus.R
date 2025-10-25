# Summarize the sdm results: model evaluation, proc time, input number of points, and variable importance ####

# Load required library
library(dplyr)

# Set input and output directories
input_dir <- "output_genus/"
output_dir <- "report_output_genus"

# Ensure output directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Loop over the group identifiers (e.g., g0–g4)
for (group in c("g0", "g1", "g2", "g3", "g4")) {
  
  message("Processing group: ", group)
  
  ### 1. Combine individual model evaluation CSVs
  csv_model_eval <- list.files(input_dir, pattern = paste0(group, "_model_evaluation\\.csv$"), full.names = TRUE)
  
  model_eval_data <- lapply(csv_model_eval, function(file) {
    df <- read.csv(file)
    df$species <- sub("_model_evaluation\\.csv", "", basename(file))
    return(df)
  })
  
  model_eval_combined <- bind_rows(model_eval_data) %>%
    filter(!is.na(calibration), !is.na(validation), !is.na(evaluation))
  
  write.csv(
    model_eval_combined,
    file.path(output_dir, paste0(group, "_combined_model_evaluation.csv")),
    row.names = FALSE
  )
  
  ### 2. Combine ensemble model evaluation CSVs
  csv_model_eval2 <- list.files(input_dir, pattern = paste0(group, "_model_evaluation2\\.csv$"), full.names = TRUE)
  
  model_eval2_data <- lapply(csv_model_eval2, function(file) {
    df <- read.csv(file)
    df$species <- sub("_model_evaluation2\\.csv", "", basename(file))
    return(df)
  })
  
  model_eval2_combined <- bind_rows(model_eval2_data)
  
  write.csv(
    model_eval2_combined,
    file.path(output_dir, paste0(group, "_combined_model_evaluation2.csv")),
    row.names = FALSE
  )
  
  ### 3. Combine processing time CSVs
  csv_proc_time <- list.files(input_dir, pattern = paste0(group, "_proc_time\\.csv$"), full.names = TRUE)
  
  proc_time_data <- lapply(csv_proc_time, function(file) {
    df <- read.csv(file)
    df$species <- sub(paste0(group, "_proc_time\\.csv"), "", basename(file))
    return(df)
  })
  
  proc_time_combined <- bind_rows(proc_time_data)
  
  write.csv(
    proc_time_combined,
    file.path(output_dir, paste0(group, "_combined_proc_time.csv")),
    row.names = FALSE
  )
  
  ### 4. Combine input summary CSVs
  csv_sum_input <- list.files(input_dir, pattern = paste0(group, "_sum_input\\.csv$"), full.names = TRUE)
  
  sum_input_data <- lapply(csv_sum_input, function(file) {
    df <- read.csv(file)
    df$species <- sub(paste0(group, "_sum_input\\.csv"), "", basename(file))
    return(df)
  })
  
  sum_input_combined <- bind_rows(sum_input_data)
  
  write.csv(
    sum_input_combined,
    file.path(output_dir, paste0(group, "_combined_sum_input.csv")),
    row.names = FALSE
  )
  
  ### 5. Combine variable importance CSVs
  csv_var_importance <- list.files(input_dir, pattern = paste0(group, "_var_importance\\.csv$"), full.names = TRUE)
  
  var_importance_data <- lapply(csv_var_importance, function(file) {
    df <- read.csv(file)
    df$species <- sub(paste0(group, "_var_importance\\.csv"), "", basename(file))
    return(df)
  })
  
  var_importance_combined <- bind_rows(var_importance_data)
  
  write.csv(
    var_importance_combined,
    file.path(output_dir, paste0(group, "_combined_var_importance.csv")),
    row.names = FALSE
  )
}