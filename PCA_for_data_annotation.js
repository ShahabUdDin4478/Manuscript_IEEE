var rectangle = 
    /* color: #d63000 */
    /* shown: false */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[70.77589184824228, 33.50149573167623],
          [70.77589184824228, 33.002502356803866],
          [71.8312659937501, 33.002502356803866],
          [71.8312659937501, 33.50149573167623]]], null, false),
    roi = 
    /* color: #98ff00 */
    /* shown: false */
    ee.Geometry.Point([71.38323974609375, 33.290827585498945]);


var sentImage = ee.ImageCollection('COPERNICUS/S2_SR')
.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 3))
.filter(ee.Filter.lt('NOT_VEGETATED_PERCENTAGE', 99))
.filterDate("2015-01-02", "2020-12-30")
.filterBounds(rectangle);
//print(sentImage)
var sentImage = sentImage.median().clip(rectangle);

//Sentinel 1 SRD
var collectionVV = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .select(['VV']);
// Create a 3 band stack by selecting from different periods (months)
var im1 = ee.Image(collectionVV.filterDate('2016-04-01', '2020-05-30').median());
var im1 = im1.clip(rectangle);
Map.addLayer(im1, {min: -25, max: 0}, 'VV stack');

var sentImage = ee.Image.cat([sentImage, im1 ]);
//Select Sentinel - 2 Bands to use in the study
var sentbands = ['B2','B3','B4','B8','B11','B12','VV'];

var NDVI = sentImage.expression(
  "(NIR-RED) / (NIR+RED)",
  {
    RED: sentImage.select("B4"),
    NIR: sentImage.select("B8"),
    BLUE: sentImage.select("B2")
  });
  
  Map.addLayer(NDVI, {min: 0, max: 1}, "NDVI");


// Nice visualization parameters for a vegetation index.
var vis = {min: 0, max: 1, palette: [
  'FFFFFF', 'CE7E45', 'FCD163', '66A000', '207401',
  '056201', '004C00', '023B01', '012E01', '011301']};
Map.addLayer(NDVI, vis, 'NDVI');
//Map.addLayer(NDVI.gt(0.5), vis, 'NDVI Binarized');


//Ignore this, this is only used to avoid errors while concatinting eigen pairs
var eigenCollection = ee.Array([[0],[1],[2],[3],[4]]);
var tscale = 5;

// Decorrelation Stretching Main Function     
function decStr(bandsImage, location, scale){
  var bandNames = bandsImage.bandNames();
  // Naming the axis for intuition
  var dataAxis = 0;
  var bandsAxis = 1;
  // Calculate the mean for each band image
  var meansAll = bandsImage.reduceRegion(ee.Reducer.mean(), location, scale);
  // Generate an array (1D Matrix) of mean of each band
  var arrayOfmeans = ee.Image(meansAll.toArray());
  // Collapse the bands data such that each pixel is a matrix of pixel values of each band
  var pixelArrays = bandsImage.toArray();
  // Use the means array and the collapsed band data to center each pixel of each band by subtracting its corresponding mean from it
  var meanCent = pixelArrays.subtract(arrayOfmeans);
  // Calculate the Covariance matrix for the bands data
  var covar = meanCent.reduceRegion({
    reducer: ee.Reducer.centeredCovariance(),
    geometry: location,
    scale: scale
  });
  
  // Get the covariance in array format which shows the band-band covarince of the data
  var covarArray = ee.Array(covar.get('array'));
  // Perform eigen decomposition of the covariance matrix to obtain eigen values and eigen vector pairs
  var eigenPairs = covarArray.eigen();
  var eigenValues = eigenPairs.slice(bandsAxis, 0, 1); // slice(axis, start, end, step)
  var eigenVectors = eigenPairs.slice(bandsAxis, 1);
  // Rotate by the eigenvectors, scale to a variance of 30, and rotate back.
  //Store a diagonal matrix in i
  var i = ee.Array.identity(bandNames.length()); // i will be used to isolate each band data and scale its variance e.g i = [1,0,0,0,0] = isolate first band from 5 bands
  // Calculate variance from the eigenvalues ---> variance = 1/sqrt(eigenvalues)
  // matrixToDiag = Computes a square diagonal matrix from a single column matrix for multiplication purposes
  var variance = eigenValues.sqrt().matrixToDiag();
  //Multiply diagonal matrix i by 30 and divide by vaiance to obtain scaling variance matrix
  var scaled = i.multiply(30).divide(variance); //Changed from 30 -> 50, It was observed that changing variance scale increases contrast. Best contrast obtained for 30
  // Calculate a rotation matrix ---> rotationMatrix =  Eigenvect.Transpose * ScaledVariance * Eigenvect
  var rotation = eigenVectors.transpose()
    .matrixMultiply(scaled)
    .matrixMultiply(eigenVectors);
  // Convert 1-D nomalized array image data to 2-D and transpose it so it can be multiplied with rotation matrix
  var transposed = meanCent.arrayRepeat(bandsAxis, 1).arrayTranspose();
  // Multiply the transposed data with the rotation matrix
  return transposed.matrixMultiply(ee.Image(rotation))
    .arrayProject([bandsAxis])   //This drop unecessary axis from the transposed data and only retains 2 axis
    .arrayFlatten([bandNames])  //Flatten collections of collections
    .add(127).byte(); // Conver pixel values to 127 means so it can be visualized between 0 - 255 range.
    
    // .byte is used to force element wise operation
}

// Principal Component Analysis Main Function
function PCA(meanCent, scale, location){
  // Flatten the band image data in from 2D to a 1D array
  var arrays = meanCent.toArray();
  //print('PCA applying on', meanCent);
  // Calculate the covariance matrix for the bands data of the region
  var covar = arrays.reduceRegion({
    reducer: ee.Reducer.centeredCovariance(),
    geometry: location,
    scale: scale,
    //tileScale: tscale,
    maxPixels: 1e9
  });
  // Get the band to band covariance of the region in 'array' format. Here .get('array') --> casts to an array
  var covarArray = ee.Array(covar.get('array'));
  // Perform an eigen analysis and slice apart the values and vectors.
  var eigenPairs = covarArray.eigen();
  // This is a P-length vector of Eigenvalues. Here P = number of PCs
  var eigenValues = eigenPairs.slice(1, 0, 1);
  // This is a PxP matrix with eigenvectors in rows.
  var eigenVectors = eigenPairs.slice(1, 1);
  //Print and store eigen pairs in eigenCollection variable and export to drive
 // print('eigen Values', eigenValues);
 // print('eigen Vector', eigenVectors);
    //Make feature collection out of eigenpairs so it can be exported to excel. From there we Convert it to a table using a python script
  eigenCollection = ee.Feature(null,{values:ee.Array.cat([eigenValues,eigenVectors],1)}); 
 // print('Eigen Collection Length',eigenCollection);
    // Export the FeatureCollection to excel sheet in drive
/*
  Export.table.toDrive({
  collection: ee.FeatureCollection([eigenCollection]),
  description: 'eigenAnalysis',
  fileFormat: 'CSV'
  });
*/
  // Convert the 1D image array back to 2D matrix for multiplication
  var imageMat = arrays.toArray(1);
  // To obtain PC = EigenVectors * 2D Image Matrix
  var PCs = ee.Image(eigenVectors).matrixMultiply(imageMat);
  // Turn the square roots of the Eigenvalues into a P-band image.
  var sdImage = ee.Image(eigenValues.sqrt())
    .arrayProject([0]).arrayFlatten([getNewBandNames('sd')]);
  // Turn the PCs into a P-band image, normalized by SD.
  return PCs
    // Throw out an an unneeded dimension, [[]] -> [].
    .arrayProject([0])
    // Make the one band array image a multi-band image, [] -> image.
    .arrayFlatten([getNewBandNames('pc')])
    // Normalize the PCs by their SDs.
    .divide(sdImage);
}

          //TCCs and FCCs

  //Sentinel - 2 L2A TCC
var trueColor = {
  bands: ["B4", "B3", "B2"],
  min: 0,
  max: 3000
};
Map.addLayer(sentImage, trueColor, "Sentinel 2 True Color");

  //Sentinel - 2 L2A FCC
var trueColor = {
  bands: ["B11", "B12", "B8"],
  min: 0,
  max: 3000
};
Map.addLayer(sentImage, trueColor, "Sentinel 2 False Color");





              //APPLYING DS ON ALL DATA
// Selecting bands to apply DS


var sentBandsImage = sentImage.select(sentbands);

//Obtain DS Results for All Satelites using dcs function


var DSsent = decStr(sentBandsImage, rectangle, 1000);

//FCC of 3 bands of DS results for all satelites
var selectBands = [0,1,2,3,4];
//Map.addLayer(DSLand.select(selectBands), {}, 'DCS Landsat Image');

var selectBands = [0,1,2,3,4,5]; 
//Map.addLayer(DSsent.select(selectBands), {}, 'DCS Sentinel Image');




            //PRINCIPAL COMPONENT ANALYSIS

  //Applying PCA on DS of Sentinel - 2
var region = sentImage.geometry();
var bands = [0,1,2,3,4,5,6];
var image =  DSsent.select(bands);
var scale = 90;
var bandNames = image.bandNames();
var meanDict = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: region,
    scale: scale,
    maxPixels: 1e9
});
var means = ee.Image.constant(meanDict.values(bandNames));
var centered = image.subtract(means);
var getNewBandNames = function(prefix) {
  var seq = ee.List.sequence(1, bandNames.length());
  return seq.map(function(b) {
    return ee.String(prefix).cat(ee.Number(b).int());
  });
};
var pcImageSDS = PCA(centered, scale, region);
/*Export.image.toDrive({
  image: pcImageDS,
  description: 'Sentinel2PCAofDS',
  folder: "GraniteExports",
  scale: 15,
  region: region
});*/

//Map.addLayer(pcImageSDS, {bands: ['pc2', 'pc3', 'pc6'], min: -2, max: 2}, 'Sentinel 2 L2C - PCA of DS (pc 2,3,6)'); //changed from PC1, PC2 and PC3


      
  //Sentinel PCA
var region = sentImage.geometry();
var image =  sentImage.select(sentbands);
var scale = 90;
var bandNames = image.bandNames();
var meanDict = image.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: region,
    scale: scale,
    maxPixels: 1e9
});
var means = ee.Image.constant(meanDict.values(bandNames));
var centered = image.subtract(means);


var pcImageS = PCA(centered, scale, region);
/*Export.image.toDrive({
  image: pcImageS,
  description: 'Sentinel2PCA',
  folder: "GraniteExports",
  scale: 15,
  region: region
});*/
//Map.addLayer(pcImageS, {bands: ['pc3', 'pc4', 'pc7'], min: -2, max: 2}, 'Sentinel 2 - PCA  used in paper 3,4,7');
/*
Export.image.toDrive({
  image: pcImageS,
  description: 'SentPCA',
  folder: "GypsumExports",
  scale: 30,
  region: region
});
*/

// Plot each PC as a new layerl
//Map.addLayer(pcImageS, {bands: ['pc2', 'pc3', 'pc4'], min: -2, max: 2}, 'Sentinel 2 - PCA  used in paper 2,3,4');

Map.addLayer(pcImageS, {bands: ['pc3', 'pc4', 'pc5'], min: -2, max: 2}, 'Sentinel 2 - PCA  used in paper 3,4,5');


  // Plot each PC as a new layer
for (var i = 0; i < bandNames.length().getInfo(); i++) {
  var band = pcImageS.bandNames().get(i).getInfo();
  Map.addLayer(pcImageS.select([band]), {min: -2, max: 2}, band);
}


Map.setCenter(71.38323974609375,33.290827585498945, 11);
