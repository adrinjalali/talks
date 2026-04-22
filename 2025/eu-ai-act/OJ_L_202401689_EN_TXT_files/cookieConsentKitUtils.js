// Cookies have been accepted ; Return true if CCK cookies have been accepted ALL
function cookiesAccepted() {
    try {
        if($wt.cookie.exists('cck1')) {
            var cck1 = JSON.parse($wt.cookie.get("cck1"));
            return (cck1.cm && cck1.all1st);
        } else {
            return false;
        }
    } catch (e) {                       //exception if $wt not exists for  setAffixSidebar() in standalone pages
        if(e instanceof ReferenceError){
            return false;
        }
    }

}
// Cookies are refused; Return true if CCK cookies have been refused.
function cookiesRefused() {
    try {
        if($wt.cookie.exists('cck1')) {
            var cck1 = JSON.parse($wt.cookie.get("cck1"));
            return (cck1.cm == true && cck1.all1st == false);
        }else {
            return false;
        }
    } catch (e){
        if (e instanceof ReferenceError ){
            return false;
        }
    }
}
// No CCK choise has been made.
function cookiesNoChoice() {
    return (cookiesAccepted()==false && cookiesRefused()==false);
}

$(document).ready(function () {
    // Send cck accepted All information to PPAS CONSENT MANAGER when user clicks the accept button of CCK.
    window.addEventListener("cck_all_accepted", function () {
        sendCckAcceptedToPpms();
    });

    // Send CCK accepted Technical Only information (sendCckRefused according to specs) to PPAS CONSENT MANAGER when user clicks the refuse button of CCK.
    window.addEventListener("cck_technical_accepted", function () {
        sendCckRefusedToPpms();
    });
    // Read the defined PIWIK pro siteIDs
    var currentSiteIds = [];
    if (typeof $("#piwikProSiteID").val() !='undefined'
        &&  $("#piwikProSiteID").val() !="") {
        currentSiteIds.push($("#piwikProSiteID").val());
    }
    if (typeof $("#piwikProSummariesSiteID").val() !='undefined'
        &&  $("#piwikProSummariesSiteID").val() !="") {
        currentSiteIds.push($("#piwikProSummariesSiteID").val());
    }
    syncCckForOtherSiteIds(currentSiteIds);
})


// Send cck refuse information to piwik pro api existing in the current page.
function sendCckAcceptedToPpms() {
    //Step 1: Javascript code to be executed when the visitor clicks on ?I accept cookies?
    if (typeof ppms !=='undefined' && typeof ppms.cm.api!=='undefined'){
        var new_consent = {consents: {}};
        new_consent.consents = {analytics: {status: 1}};
        ppms.cm.api('setComplianceSettings', new_consent, function (new_consent) {
            console.log("cck accepted has been sent to ppms: ");
            console.log(new_consent);
        });
    } else {
        console.log('ppms is undefined');
    }
    setAffixSidebar();
}

// Send cck refuse information to piwik pro api existing in the current page.
function sendCckRefusedToPpms() {
    if (typeof ppms !=='undefined' && typeof ppms.cm.api!=='undefined'){
        //Step 2: Javascript code to be executed when the visitor clicks on ?I refuse cookies?
        var new_consent = {consents: {}};
        new_consent.consents = {analytics: {status: 0}};
        ppms.cm.api('setComplianceSettings', new_consent, function (new_consent) {
            console.log("cck refused has been sent to ppms: ");
            console.log(new_consent);
        });
    } else {
        console.log('ppms is undefined');
    }
    setAffixSidebar();
}

// Send cck no decision information to piwik pro api existing in the current page.
function sendNoDecisionToPpms() {
    if (typeof ppms !=='undefined' && typeof ppms.cm.api!=='undefined'){

        //EURLEXNEW-4468
        var initialConsent = {consents: ['analytics']};
        ppms.cm.api('setInitialComplianceSettings', initialConsent, function(initialConsent){
            console.log("cck no decision has been sent to ppms: ");
            console.log(initialConsent);
        });

    } else {
        console.log('ppms is undefined');
    }
}

// Send the respective cck decision to piwik pro if it is not already
// set to a status that corresponds to the current CCK decision state (accept/refuse/not answered).
// Note: When clicking consent/no consent, the request is only sent for the piwik pro tag existing in the current page (e.g summaries tag). So the status of PPAS manager for other subsites,
// will be updated with this methods for all unset sites when they are visited (e.g navigate from summaries to homepage).

function syncCckForOtherSiteIds(currentSiteIds) {
    for (var i = 0; i < currentSiteIds.length; i++) {
        if (currentSiteIds[i] != "") {
            var cookieKey = "ppms_privacy_" + currentSiteIds[i];
            var ppmsCookie = readCookie(cookieKey);
            if (ppmsCookie != "") {
                var cookieValueJson = JSON.parse(ppmsCookie);
                var status = cookieValueJson.consents.analytics.status;
                if (cookiesAccepted() && status != 1) {
                    console.log ("Apply consent for siteID: " + cookieKey);
                    sendCckAcceptedToPpms();
                }
                else if (cookiesRefused() && status != 0) {
                    console.log ("Apply no consent for siteID: " + cookieKey);
                    sendCckRefusedToPpms();
                }
                else if (cookiesNoChoice() && status != -1) {
                    console.log ("Would Apply no decision for siteID: " + cookieKey);
                    sendNoDecisionToPpms();
                }
            }

        }
    }
}
