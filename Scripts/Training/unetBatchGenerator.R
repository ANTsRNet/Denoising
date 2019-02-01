unetImageBatchGenerator <- function( batchSize = 32L,
                                     stepsPerEpoch = 48L,
                                     patchSize = c( 32, 32, 32 ),
                                     maxNumberOfPatchesPerSubject = 2,
                                     segmentationLabels = NA,
                                     segmentationLabelWeights = NA,
                                     reorientImage = NA,
                                     originalList = NULL,
                                     denoisedList = NULL,
                                     segmentationList = NULL )
{

  if( is.null( originalList ) )
    {
    stop( "Original images must be specified." )
    }
  if( is.null( denoisedList ) )
    {
    stop( "Denoised images must be specified." )
    }
  if( is.null( segmentationList ) )
    {
    stop( "Input segmentation images must be specified." )
    }

  currentPassCount <- 0L

  function()
    {

    shuffledIndices <- sample.int( length( originalList ) )

    batchOriginals <- originalList[shuffledIndices]
    batchDenoiseds <- denoisedList[shuffledIndices]
    batchSegmentations <- segmentationList[shuffledIndices]

    channelSize <- length( batchOriginals[[1]] )

    batchX <- array( data = 0, dim = c( batchSize, patchSize, channelSize ) )
    batchY <- array( data = 0, dim = c( batchSize, patchSize, 1 ) )

    i <- 1
    while( i < batchSize )
      {
      subjectSegmentation <- antsImageRead( batchSegmentations[[i]] )
      subjectSegmentation[subjectSegmentation != 0] <- 1

      subjectDenoised <- antsImageRead( batchDenoiseds[[i]] )
      subjectDenoised[subjectSegmentation == 0] <- 0

      subjectDenoisedArray <- as.array( subjectDenoised )

      imageSd <- sd( subjectDenoisedArray[which( subjectDenoisedArray != 0 )] )
      imageMean <- mean( subjectDenoisedArray[which( subjectDenoisedArray != 0 )] )
      subjectDenoisedArray[which( subjectDenoisedArray != 0 )] <-
        ( subjectDenoisedArray[which( subjectDenoisedArray != 0 )] - imageMean ) / imageSd

      subjectDenoised <- as.antsImage( subjectDenoisedArray )

      randomSeed <- sample( 1:2^15, 1 )

      subjectDenoisedPatches <- extractImagePatches( subjectDenoised,
        patchSize, maxNumberOfPatches = maxNumberOfPatchesPerSubject,
        maskImage = subjectSegmentation, randomSeed = randomSeed,
        returnAsArray = TRUE )

      subjectOriginals <- batchOriginals[[i]]

      subjectOriginalPatches <- list()
      for( k in seq_len( channelSize ) )
        {
        subjectChannelImage <- antsImageRead( subjectOriginals[k] )
        subjectChannelImage[subjectSegmentation == 0] <- 0
        subjectChannelArray <- as.array( subjectChannelImage )

        imageSd <- sd( subjectChannelArray[which( subjectChannelArray != 0 )] )
        imageMean <- mean( subjectChannelArray[which( subjectChannelArray != 0 )] )
        subjectChannelArray[which( subjectChannelArray != 0 )] <-
          ( subjectChannelArray[which( subjectChannelArray != 0 )] - imageMean ) / imageSd

        subjectChannelImage <- as.antsImage( subjectChannelArray )

        subjectOriginalPatches[[k]] <- extractImagePatches( subjectChannelImage,
          patchSize, maxNumberOfPatches = maxNumberOfPatchesPerSubject,
          maskImage = subjectSegmentation, randomSeed = randomSeed,
          returnAsArray = TRUE )
        }
      for( j in seq_len( dim( subjectDenoisedPatches )[1] ) )
        {
        batchY[i,,,,1] <- subjectDenoisedPatches[j,,,]
        for( k in seq_len( channelSize ) )
          {
          patchArray <- subjectOriginalPatches[[k]][j,,,]
          batchX[i,,,,k] <- patchArray
          }
        i <- i + 1

        if( i > batchSize )
          {
          break
          }
        }
      }

    if ( currentPassCount == ( stepsPerEpoch - 1 ) )
      {
      currentPassCount <<- 0
      }
    else
      {
      currentPassCount <<- currentPassCount + 1
      }

    return( list( batchX, batchY ) )
    }
}