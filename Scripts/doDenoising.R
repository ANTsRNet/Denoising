library( ANTsR )
library( ANTsRNet )
library( keras )

# args <- commandArgs( trailingOnly = TRUE )

# if( length( args ) != 3 )
#   {
#   helpMessage <- paste0( "Usage:  Rscript doDenoising.R inputFile inputMaskFile outputFile\n" )
#   stop( helpMessage )
#   } else {
#   inputFileName <- args[1]
#   inputMaskFileName <- args[2]
#   outputFileName <- args [3]
#   }

inputFileName <- "Data/Example/1097782_defaced_MPRAGE.nii.gz"
inputMaskFileName <- "Data/Example/1097782_defaced_MPRAGEBrainExtractionMask.nii.gz"
outputFileName <- "denoisedOutput.nii.gz"

patchSize <- c( 32, 32, 32 )
numberOfFiltersAtBaseLayer <- 32
strideLength <- c( 16, 16, 16 )

imageMods <- c( "T1" )
channelSize <- length( imageMods )

unetModel <- createUnetModel3D( c( patchSize, channelSize ),
  numberOfOutputs = 1, dropoutRate = 0.0,
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = numberOfFiltersAtBaseLayer,
  convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
  weightDecay = 1e-5, mode = "regression" )

cat( "Loading weights file" )
startTime <- Sys.time()
weightsFileName <- paste0( getwd(), "/n4DenoiseWeights.h5" )
if( ! file.exists( weightsFileName ) )
  {
  weightsFileName <- getPretrainedNetwork( "denoising", weightsFileName )
  }
load_model_weights_hdf5( unetModel, filepath = weightsFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

# Process input

startTimeTotal <- Sys.time()

cat( "Reading ", inputFileName )
startTime <- Sys.time()
imageOriginal <- antsImageRead( inputFileName, dimension = 3 )
mask <- antsImageRead( inputMaskFileName, dimension = 3 )
mask[mask != 0] <- 1
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Extracting patches based on mask." )
startTime <- Sys.time()

imageArray <- as.array( imageOriginal )

imageSd <- sd( imageArray[which( imageArray != 0 )] )
imageMean <- mean( imageArray[which( imageArray != 0 )] )
imageArray[which( imageArray != 0 )] <-
  ( imageArray[which( imageArray != 0 )] - imageMean ) / imageSd

image <- as.antsImage( imageArray )

imagePatches <- extractImagePatches( image, strideLength = strideLength,
  patchSize, maxNumberOfPatches = 'all', maskImage = mask,
  returnAsArray = TRUE )

batchX <- array( data = 0, dim = c( dim( imagePatches )[1], patchSize, 1 ) )
batchX[,,,,1] <- imagePatches

endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Prediction and decoding" )
startTime <- Sys.time()
predictedDataArray <- unetModel %>% predict( batchX, verbose = 1 )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( " (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Reconstruct from patches" )
startTime <- Sys.time()

cleanedImage <- reconstructImageFromPatches( drop( predictedDataArray ),
  mask, strideLength = strideLength, domainImageIsMask = TRUE )

antsImageWrite( cleanedImage, outputFileName )

endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

endTimeTotal <- Sys.time()
elapsedTimeTotal <- endTimeTotal - startTimeTotal
cat( "\nTotal elapsed time:", elapsedTimeTotal, "seconds\n\n" )
