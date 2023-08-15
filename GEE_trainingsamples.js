//==================================================
// Download Sentinel-2 images and Segmentation Masks
//==================================================

// Get the list of AOIs
var aois = bb.geometry();
var segs = seg.geometry();

// Loop through the list of AOIs and download the images for each AOI
for (var i = 0; i < 20 ; i++) {
  var aoi = aois.coordinates().get(i);
  downloadImageForAOI(aoi, i);
}

for (var i = 0; i < 20 ; i++) {
  var segMask = segs.coordinates().get(i);
  var aoi = aois.coordinates().get(i);
  createSegMask(segMask, aoi, i);
}

//===========
// Functions
//===========

// Function to mask the cloud
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

// Function to download images for each AOI
function downloadImageForAOI(aoi, index) {
  print("Processing Image " + index);

  // Convert the AOI to an EE Geometry Polygon
  var aoiPolygon = ee.Geometry.Polygon(aoi);

  // Filter the Sentinel-2 ImageCollection for the AOI and other criteria
  var dataset = ee.ImageCollection('COPERNICUS/S2')
    .filterDate('2021-01-01', '2021-12-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
    .filterBounds(aoiPolygon)
    .map(maskS2clouds)
    .map(function(image) { return image.clip(aoiPolygon); });

  // Select the required bands
  var required_bands = ['B2', 'B3', 'B4'];
  dataset = dataset.median().select(required_bands);

  // Visualization of layer to the map
  var visualization = {
    min: 0.0,
    max: 0.3,
    bands: ['B4', 'B3', 'B2'],
  };
  
  // Set the map center to the AOI and add the dataset as a layer
  Map.centerObject(aoiPolygon);
  Map.addLayer(dataset, visualization, 'RGB');

  // Export the image to Google Drive
  Export.image.toDrive({
    image: dataset,
    description: 'img_' + index, // Adding index to the description to differentiate between AOIs
    region: aoiPolygon,
    dimensions: [448, 448]
  });
}

// Function to create an image with pixels set to 0 everywhere except inside the specified polygon where pixel values are 1
function createSegMask(segMask, aoi, index) {
  print("Processing Mask " + index);

  // Convert the AOI to an EE Geometry Polygon
  var segMaskPolygon = ee.Geometry.Polygon(segMask);
  var aoiPolygon = ee.Geometry.Polygon(aoi);
  
   // Create an image with constant value 1 inside the segMaskPolygon and 0 outside
  var maskImage = ee.Image.constant(0).clip(aoiPolygon)
    .paint(segMaskPolygon, 1);

  // Set the map center to the segMaskPolygon and add the image as a layer
  Map.centerObject(maskImage);
  Map.addLayer(maskImage, { min: 0, max: 1, palette: ['black', 'white'] }, 'Segmentation Mask');
  

 // Export the image to Google Drive
  Export.image.toDrive({
    image: maskImage,
    description: 'mask_' + index, // Adding index to the description to differentiate between AOIs
    region: aoiPolygon,
    dimensions: [448, 448]
  });
  
}
