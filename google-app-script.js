function onEdit(e) {
  const value = callMyFastAPI(); 
  console.log(value)
  console.log("trigger")
}



function callMyFastAPI() {
  const url = "https://baseline-barrier-layout-quiet.trycloudflare.com/trigger"; 
  const options = {
    method: "get",
    muteHttpExceptions: false
  };

  try {
    const response = UrlFetchApp.fetch(url, options);
    return response.getContentText();
  } catch (err) {
    return "Error: " + err.toString();
  }
}
