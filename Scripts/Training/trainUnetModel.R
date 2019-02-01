library( ANTsR )
library( ANTsRNet )
library( keras )
library( tensorflow )

patchSize <- c( 32, 32, 32 )
numberOfFiltersAtBaseLayer <- 32
strideLength <- c( 16, 16, 16 )
maxNumberOfPatchesPerSubject <- 100
batchSize <- 32L

keras::backend()$clear_session()

imageMods <- c( "T1" )
channelSize <- length( imageMods )

baseDirectory <- '/home/ntustison/Data/'
scriptsDirectory <- paste0( baseDirectory, 'Scripts/Denoising/' )
source( paste0( scriptsDirectory, 'unetBatchGenerator.R' ) )

dataDirectories <- c()
dataDirectories <- append( dataDirectories, paste0( baseDirectory, "ADNI/" ) )
dataDirectories <- append( dataDirectories, paste0( baseDirectory, "CorticalThicknessData2014/IXI/ThicknessAnts/" ) )
dataDirectories <- append( dataDirectories, paste0( baseDirectory, "CorticalThicknessData2014/Kirby/ThicknessAnts/" ) )
dataDirectories <- append( dataDirectories, paste0( baseDirectory, "CorticalThicknessData2014/NKI/ThicknessAnts/" ) )
dataDirectories <- append( dataDirectories, paste0( baseDirectory, "CorticalThicknessData2014/Oasis/ThicknessAnts/" ) )

brainImageFiles <- c()
for( i in seq_len( length( dataDirectories ) ) )
  {
  imageFiles <- list.files( path = dataDirectories[i],
    pattern = "*BrainSegmentation0N4Denoised.nii.gz", full.names = TRUE, recursive = TRUE )
  brainImageFiles <- append( brainImageFiles, imageFiles )
  }

trainingImageFiles <- list()
trainingOriginalFiles <- list()
trainingSegmentationFiles <- list()

missingFiles <- c()

cat( "Loading data...\n" )
pb <- txtProgressBar( min = 0, max = length( brainImageFiles ), style = 3 )

count <- 1
for( i in seq_len( length( brainImageFiles ) ) )
  {
  setTxtProgressBar( pb, i )

  brainImageFile <- brainImageFiles[i]
  subjectDirectory <- dirname( brainImageFiles[i] )
  subjectId <- basename( brainImageFile )
  subjectId <- sub( "BrainSegmentation0N4Denoised.nii.gz", '', subjectId )

  if( grepl( "LongitudinalThicknessANTsNative", subjectDirectory ) )
    {
    brainOriginalFile <- paste0( subjectDirectory, "/", subjectId, "0.nii.gz" )
    } else {
    t1Directory <- sub( "ThicknessAnts", 'T1', subjectDirectory )
    t1Files <- list.files( t1Directory, pattern = paste0( subjectId, "*" ), full.names = TRUE )

    brainOriginalFile <- t1Files[1]
    }
  brainSegmentationFile <- paste0( subjectDirectory, "/", subjectId, "BrainSegmentation.nii.gz" )

  missingFile <- FALSE

  if( ! file.exists( brainSegmentationFile ) )
    {
    stop( paste( "File does not exist ---> ", brainSegmentationFile, "\n" ) )
    missingFile <- TRUE
    }

  if( ! file.exists( brainOriginalFile ) )
    {
    stop( paste( "File does not exist ---> ", brainOriginalFile, "\n" ) )
    missingFile <- TRUE
    }

  if( missingFile )
    {
    missingFiles <- append( missingFiles, brainImageFile )
    } else {
    trainingImageFiles[[count]] <- brainImageFile
    trainingOriginalFiles[[count]] <- brainOriginalFile
    trainingSegmentationFiles[[count]] <- brainSegmentationFile
    count <- count + 1
    }
  }
cat( "\n" )


###
#
# Create the Unet model
#

# See this thread:  https://github.com/rstudio/tensorflow/issues/272

# with( tf$device( "/cpu:0" ), {
unetModel <- createUnetModel3D( c( patchSize, channelSize ),
  numberOfOutputs = 1, dropoutRate = 0.0,
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = numberOfFiltersAtBaseLayer,
  convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
  weightDecay = 1e-5, mode = "regression" )
  # } )

# if( file.exists( paste0( scriptsDirectory, "/n4DenoiseWeights.h5" ) ) )
#   {
#   load_model_weights_hdf5( unetModel, paste0( scriptsDirectory, "/n4DenoiseWeights.h5" ) )
#   }

unetModel %>% compile( loss = "mean_squared_error",
  optimizer = optimizer_adam( lr = 0.0001 ),
  metrics = c( "mean_squared_error" ) )

###
#
# Set up the training generator
#

# Split trainingData into "training" and "validation" componets for
# training the model.

numberOfData <- length( trainingImageFiles )
sampleIndices <- sample( numberOfData )

validationSplit <- floor( 0.8 * numberOfData )
trainingIndices <- sampleIndices[1:validationSplit]
numberOfTrainingData <- length( trainingIndices )
validationIndices <- sampleIndices[( validationSplit + 1 ):numberOfData]
numberOfValidationData <- length( validationIndices )

###
#
# Run training
#

track <- unetModel %>% fit_generator(
  generator = unetImageBatchGenerator( batchSize = batchSize,
                                       stepsPerEpoch = 32L,
                                       patchSize = patchSize,
                                       maxNumberOfPatchesPerSubject = maxNumberOfPatchesPerSubject,
                                       originalList = trainingOriginalFiles[trainingIndices],
                                       denoisedList = trainingImageFiles[trainingIndices],
                                       segmentationList = trainingSegmentationFiles[trainingIndices]
                                     ),
  steps_per_epoch = 32L,
  epochs = 500L,
  validation_data = unetImageBatchGenerator( batchSize = batchSize,
                                             stepsPerEpoch = 16L,
                                             patchSize = patchSize,
                                             maxNumberOfPatchesPerSubject = maxNumberOfPatchesPerSubject,
                                             originalList = trainingOriginalFiles[validationIndices],
                                             denoisedList = trainingImageFiles[validationIndices],
                                             segmentationList = trainingSegmentationFiles[validationIndices],
                                           ),
  validation_steps = 16L,
  callbacks = list(
    callback_model_checkpoint( paste0( scriptsDirectory, "/n4DenoiseWeights.h5" ),
      monitor = 'val_loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto', period = 1 ),
     callback_reduce_lr_on_plateau( monitor = 'val_loss', factor = 0.5,
       verbose = 1, patience = 10, mode = 'auto' ),
     callback_early_stopping( monitor = 'val_loss', min_delta = 0.0001,
       patience = 50 )
  )
)

save_model_weights_hdf5( unetModel, paste0( scriptsDirectory, "/n4DenoiseWeights.h5" ) )

