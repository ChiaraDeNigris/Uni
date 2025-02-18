library(reticulate)
library(rjson)
library(jpeg)
library(grid)
library(VGAM)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Function definition 
# Split the data into random calibration and validation set
cal_val_split <- function(n, smx, labels, seed = NULL){ 
  if(!is.null(seed)){
    set.seed(seed)
  } else { 
    set.seed(seed)
  }
  f_idx <- c(rep(1, n), rep(0, nrow(smx) - n)) > 0
  f_idx <- sample(f_idx) 
  f_cal_smx <- smx[f_idx,] 
  f_cal_labels <- labels[f_idx]
  f_val_smx <- smx[!f_idx,]
  f_val_labels <- labels[!f_idx]
  return(list(idx = f_idx, cal_smx = f_cal_smx, cal_labels = f_cal_labels, val_smx = f_val_smx, val_labels = f_val_labels))
}

# Calculate the qhat
q_function <- function(alpha, n, cal_smx, cal_labels, cal_scores){
  # score function = 1 - softmax output
  cal_scores <- 1 - cal_smx[cbind(1:n, cal_labels + 1)] # conformal score value of the true class for each record
  q_level <- ceiling((n + 1) * (1 - alpha)) / n # adjusted quantile for finite sample
  qhat <- quantile(cal_scores, probs = q_level, type = 1, na.rm = TRUE) 
  return(qhat)
}

np <- import("numpy")

# Load model prediction
data <- np$load("imagenet-resnet152.npz") 

smx <- data['smx'] # softmax score
labels <- as.integer(data['labels']) 

n <- 2500 # number of calibration points
alpha <- 0.1 # 1-alpha is the desired coverage

# Testing the performances of the Conformal Prediction
coverage <- vector() 
sets <- vector()
adaptability <- vector()

for (t_run in 1:100) {
  # create random calibration and validation sets
  splits <- cal_val_split(n, smx, labels, seed = t_run)
  
  # create prediction set for each run
  qhat <- q_function(alpha, n, splits$cal_smx, splits$cal_labels, splits$cal_scores) 
  # boolean matrix of  TRUE/ FALSE if  score > di 1 - qhat
  prediction_sets <- splits$val_smx >= (1 - qhat)
  
  # calculate the metrics
  covered <- prediction_sets[cbind(1:nrow(prediction_sets), splits$val_labels + 1)]
  coverage <- c(coverage, mean(covered))
  sets <- c(sets, rowSums(prediction_sets))
  adaptability <- rbind(adaptability, sapply(0:(dim(smx)[2]- 1), function(label) {mean(covered[splits$val_labels == label])}))
}

# Define betabinomial distribution 
nval <- length(splits$val_labels)
l <- floor((n + 1) * alpha)
a <- n + 1 - l
b <- l
x <- floor(nval * 0.75):nval
rv <- dbetabinom.ab(x, size = nval, shape1 = a, shape2 = b) #define the distribution

# Plot coverage
par(mar = c(5, 5, 3, 5))

hist(coverage, main = "Coverage distribution", xlim = c(0.88, 0.92), ylim= c(0, 100),
     xlab = "Coverage", col = "lightblue", border = "white", freq = FALSE)
abline(v = 1 - alpha, col = "black", lwd = 2) #add a line for 1-alpha

par(new = TRUE) 

# Plot betabinomial distribution
plot(x / nval, rv * nval / 100, type = "l", lwd = 3, col = "red", ylab = "", xlab = "", 
     axes = FALSE, ylim = c(0, 1), xlim = c(0.88, 0.92))

axis(side = 4, lwd = 1, las = 1)
mtext("Betabinomial Probability Density Function", side = 4, line = 3, cex = 1)
legend("topright", legend = c("Empirical", paste("1 - alpha: ", 1-alpha), "Theoretical"),
       col = c("lightblue","black", "red"), lty = c(1, 1), lwd = c(1, 3), cex = 0.6)

# Plot set size
for_barplot <- table(sets) 
barplot(for_barplot, names.arg= names(for_barplot), xlab = "Sets Size", col = "lightblue", border = "white", main = "Sets Size")
legend("topright", paste("The average set size is:", round(mean(sets), 2)))

barplot(for_barplot, names.arg= names(for_barplot), log = "y", xlab = "Sets Size", col = "lightblue", border = "white", main = "Sets Size with y in log scale")
legend("topright", paste("The average set size is:", round(mean(sets), 2)))

# Plot adaptability - each bar of the bar plot is a single label
par(mar = c(5, 5, 3, 1))

# adaptability at the end of the loop is a matrix with 1000 columns and 100 rows, 
# each column corresponds to a label and each row corresponds to a loop execution 
for_plot <- colMeans(adaptability) # calculate the average coverage of each class
barplot(for_plot, border = "lightblue",xlab = "Label", ylab = "Coverage", main = "Labels Coverage")

abline(h = 1-alpha, col = "black", lty = "dotted")
legend("right", legend = paste("1 - alpha: ", 1-alpha),
       col = "black", lty = c(1, 1), lwd = c(1, 3), cex = 0.8)
mtext(paste("The average coverage is", round(mean(for_plot), 2)*100, "%"), side = 1)

# To better understand the above plot we replicate the bar plot but only with the first 50 labels
barplot(for_plot[1:50], names.arg=0:49, border = "white", col = "blue",xlab = "Label", ylab = "Coverage", main = "First 50 labels coverage")
abline(h = 1-alpha, col = "black")
legend("right", legend = paste("1 - alpha: ", 1-alpha),
       col = "black", lty = c(1, 1), lwd = c(1, 3), cex = 0.6)


#Example of Conformal Prediction

# Split in calibration and validation
split <- cal_val_split(n, smx, labels)

# Calculate the qhat
qhat <- q_function(alpha, n, split$cal_smx, split$cal_labels, split$cal_scores) 
qhat

# Form prediction sets
prediction_sets <- split$val_smx >= (1 - qhat)

# Calculate empirical coverage
empirical_coverage <- mean(prediction_sets[cbind(1:nrow(prediction_sets), split$val_labels + 1)])
cat("The empirical coverage is:", round(empirical_coverage, 2), "\n")

# Read label strings from JSON file
label_strings <- fromJSON(file='human_readable_labels.json')

# Get list of example image paths
example_paths <- list.files(path = 'examples', full.names = TRUE) 

# Loop over a range of 10 examples
for (i in 1:10) {
  rand_path <- sample(example_paths, 1) # randomly select an example path
  img <- jpeg::readJPEG(rand_path) # read the image
  img_index <- as.integer(tools::file_path_sans_ext(basename(rand_path))) # extract image index from the file name
  
  # create the prediction set
  prediction_set <- smx[img_index, ] >= (1 - qhat) 
  
  # display the image
  grid::grid.newpage()
  grid::grid.raster(img, interpolate = FALSE)
  
  # print the prediction set
  cat("The prediction set is: {", paste(label_strings[prediction_set], collapse = ", "), "}\nThe true label is:", label_strings[labels[img_index] + 1], "\n\n")
}