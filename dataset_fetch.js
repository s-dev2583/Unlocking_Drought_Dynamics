// ------------------------------------------------- //
//       EXPORT GEE TABLE ASSET TO GOOGLE DRIVE       //
// ------------------------------------------------- //

// ✅ Replace with your table asset ID
var tableAssetId = 'projects/ee-semproject/assets/Maharashtra_VCI_TCI_SAR_2015_2024_Full';

// ✅ Load the table asset
var tableAsset = ee.FeatureCollection(tableAssetId);

// ✅ Export the table to Google Drive
Export.table.toDrive({
    collection: tableAsset,
    description: 'Maharashtra_Drought_2023_2024_Full_Drive',
    folder: 'GEE_Exports',              // ✅ Destination folder in Google Drive
    fileFormat: 'CSV'                   // ✅ File format
});

print("Exporting table asset to Google Drive...");
