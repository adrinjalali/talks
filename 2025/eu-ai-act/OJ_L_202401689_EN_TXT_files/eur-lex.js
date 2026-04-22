var dateLineTemplate ="";
var maxNbDayForCookie = 30;
var lastFocusElement = null;
var exportHref = null;

if (typeof(legislativeUrl)!='undefined') {
	var toolbox = '<div class="toolsList"><a href="'+legislativeUrl+'">'+legDraftLabel+'<\/a> <a href="'+eurovocUrl+'">Eurovoc<\/a><a href="'+interStyleGuideUrl+'">'+interStyleGuideLabel+'<\/a><a href="'+otherLink+'">'+otherLinksLabel+'<\/a><\/div>';
}

$(document).ready(function(){
	//polyfill for IE11 to support startsWith()
	if (!String.prototype.startsWith) {
		String.prototype.startsWith = function(searchString, position){
			position = position || 0;
			return this.substr(position, searchString.length) === searchString;
		};
	}

	//EURLEXNEW-4006: Function to adjust summaries topic TOC based on available vertical space.
	$(window).on("load resize scroll", function () {
		if ($("#summariesByTopic").length && window.innerWidth > 991) {	//desktop resolution
			$("#summariesByTopic").css("height", ""); //clear previous style
			var availableHeight;
			var maxHeight = $(window).height() - $("#AffixSidebar").height() - 50;
			//Calculate if footer is in view. Copied from TOC.js
			var footerTop = $("footer").offset().top;
			var footerBottom = footerTop + $("footer").outerHeight();
			var viewportTop = $(window).scrollTop();
			var viewportBottom = viewportTop + $(window).height();
			var footerInViewPort = (footerBottom > viewportTop && footerTop < viewportBottom);

			if (footerInViewPort) { //When footer is in view
				availableHeight = $("footer").offset().top - $("#summariesByTopic").offset().top - 20;
			} else if ($("#AffixSidebar").hasClass("affix-top")) { //When sidebar is affixed on top of page
				availableHeight = maxHeight - $("#AffixSidebar").offset().top + $(window).scrollTop();
			} else { //Middle of the page.
				availableHeight = maxHeight;
			}
			$("#summariesByTopic").css("height", availableHeight);
		}
	});

	docReady();
	// Create user msg if IE<11 only in homepage and save preference after dismissal in cookie to not display again in current session
	createUserMsgIncompatibleBrowser();
	//set global var to store href
	exportHref = saveHrefExport().attr("href");
	// Listener for link to load function
	saveHrefExport().click(function(e){
		if (exportHref != null) checkAndExportSelection(e);
	});

	//Export all link inSearch results
	//Any shown user message should be hidden
	$('#link-export-documentAllTop, #link-export-documentAllBottom').on('click', function(e){
		$(".alert-warning").addClass("hidden");
		$(".alert-success").addClass("hidden");
		$(".alert-info").addClass("hidden");
	});

	/* Widget expand/collapse border line*/
	$("#searchWidgetId").click(function(){
		$(this).focus();
	});

	/*$("#recentlyPubishedId").click(function(){
		$(this).focus();
	});
	$("#recentlyPublishedNumItems").click(function(){
		$(this).focus();
	});*/

	/*contact-attach border line*/
	$("#attachFileId").click(function(){
		$(this).addClass("borderFocus");
	});
	$("#attachFileId").on('keyup', function(){
		document.getElementById("attachFileId").value = "007";
		$(this).addClass("borderFocus");
	});


	/*see-also widget border line*/
	/*$("#control_suggestedLegalContentTabContainer").click(function(){
		$(this).focus();
	});*/
	/*	$("#suggestedNumItems").click(function(){
            $(this).focus();
        });*/

	/*news widget border line*/
	/*	$("#newsWidgetId").click(function(){
            $(this).focus();
        });
        $("#newsNumItems").click(function(){
            $(this).focus();
        });
        $("#browseId").on('keyup click', function(){
            $(this).addClass("borderFocus");
        });*/


	//Advanced search form - procedure status
	$("#dateToWithdrawn,#dateFromWithdrawn,#dateExactWithdrawn,#dateToAdopted,#dateFromAdopted,#dateExactAdopted").focus(function() {
		radioStatus = this.value;
	});

	//Declaration of the needed variables
	var procStatusAdd = $("#procStatusAdopted");
	var procStatusStop = $("#procStatusWithdrawn");
	var dateExact=$("#dateExactAdopted, #dateExactWithdrawn");
	var dateFrom=$("#dateFromAdopted, #dateFromWithdrawn");
	var dateTo=$("#dateToAdopted, #dateToWithdrawn");
	var procStatus=$("#procStatusPending , #procStatusAll ");

	//Function to select which calendar will be displayed
	function calendarDisplay(calendarAdopted, calendarWithdrawn) {
		$(calendarAdopted).removeClass("hidden");
		$(calendarWithdrawn).addClass("hidden");
	}

	//Function to select if the calendar will be disabled
	function calendarDisable(value) {
		dateExact.prop("disabled", value);
		dateFrom.prop("disabled", value);
		dateTo.prop("disabled", value);
	}
	//Initialize the calendar status depending on which radio button is checked
	if (procStatusAdd.is(":checked")) {
		calendarDisplay(adoptedProc, withdrawnProc);
	}

	if (procStatusStop.is(":checked")) {
		calendarDisplay(withdrawnProc, adoptedProc);
	}

	//Disable the calendar when All or Ongoing procedure is checked
	if (procStatus.is(":checked")) {
		calendarDisplay(adoptedProc, withdrawnProc);
		calendarDisable(true);
	}

	//Disable the calendar when the selected option changes to All or Ongoing procedure
	$(procStatus).on('change', function() {
		if (procStatus.is(":checked")) {
			calendarDisable(true);
		}
	});

	//Enable the required calendar when the selected option changes to the Completed procedure
	$(procStatusAdd).on('change', function() {
		calendarDisplay(adoptedProc, withdrawnProc);
		if (procStatusAdd.is(":checked")) {
			calendarDisable(false);
		}
	});

	//Enable the required calendar when the selected option changes to the Stopped procedure
	$(procStatusStop).on('change', function() {
		calendarDisplay(withdrawnProc, adoptedProc);
		if (procStatusStop.is(":checked")) {
			calendarDisable(false);
		}
	});
});

$(window).on('load',function () {
	guidedTourTagging();
});



//	reset other fields than ALL
function resetAllStatusFields() {
	resetFields(['lpRespService','lpAdoptedAct','dateExactAdopted','dateFromAdopted','dateToAdopted','dateExactWithdrawn','dateFromWithdrawn','dateToWithdrawn']);
}


// reset other fields than PENDING
function resetPendingStatusFields() {
	resetFields(['lpAdoptedAct','dateExactAdopted','dateFromAdopted','dateToAdopted','dateExactWithdrawn','dateFromWithdrawn','dateToWithdrawn']);
}

// reset other fields than ADOPTED
function resetAdoptedStatusFields() {
	resetFields(['lpRespService','dateExactWithdrawn','dateFromWithdrawn','dateToWithdrawn']);
}

// reset other fields than WITHDRAWN
function resetWithDrawnStatusFields() {
	resetFields(['lpRespService','lpAdoptedAct','dateExactAdopted','dateFromAdopted','dateToAdopted']);
}

// Catch the pageshow event that allows performing actions even when the page is loaded from cache.
// Fixes the issue of firefox caching the overlay of the spinner
// and blocking the page when going back.
// See EURLEXNEW-3490.
$(window).on("pageshow", function (event) {
	hideHourglass();
});

function docReady() {
	/*	if (navigator.userAgent.indexOf('MSIE') == -1){
            // ELX-2041 -> Gif is not displayed while the page is reloading when
            // submitting a form, this will preload it. Known browser issue for
            // Google Chrome and Mozilla Firefox.
            $("body").prepend('<img style="position:absolute; top:-1000px;" src="' + imageMap['ajax-loader'] + '" name="load" id="load" alt="Loading, please wait..." />');
        }*/

	//Initialize the collapse panel on-load in mobile devices
	initilizeCollapsableElements();
	//Initialize the behavior of mutually exclusive panels (widgets) with/without cookies
	initializeMutuallyExclusizeCollapsablePanels();

	// used in quick-search-simple.tag
	$("#freetextEditorial,#freetext").focus(function(){
		if (navigator.userAgent.indexOf('MSIE 7')!=-1 || navigator.userAgent.indexOf('MSIE 8')!=-1){
			$(this).data('placeholder').addClass('placeholder-hide-except-screenreader');
		}
	}).change(function(){
		showPlaceholderIfEmpty($(this));
	});

	// Set timeout for session expiration
	if (typeof(sessionTimeout)!='undefined' && sessionTimeout != -1) {
		setTimeout(function() {
			sessionExpired = true;
		}, sessionTimeout);
	}


	$("a.noJsCancel").each( function(){ $(this).before("<input style='"+$(this).attr('style')+"' type='button' class='button' value=\""+$(this).text()+"\" onclick='window.location=\""+this.href+"\"' />"); $(this).remove(); });


	$(document).on('click','.simpleJsTree a.expandLink', function() {
		toggleNextTreeLevel(this)
	});
	$("a.advancedSearchTitleLink").click(function() { $(this).parent().find(".widgetControl a").first().click()});


	// Attaches a modal opening event where the eurlexModal class exists
	$(document).on('click','.eurlexModal', function() {
		var callback = $(this).data('callback');
		var url = this.href;
		showModal(url, callback);
		return false;
	});

	$(document).on('click','.authentication_required', function() {
		var callback = $(this).data('callback');
		var url = this.href;
		if(sessionExpired) {
			showAuthenticationRequiredModal(url, callback);
			return false;
		}
	});

	// TODO Is this needed any more
	$(".imgWithHover").hover(
		function() {
			var img = $(this).attr("src");
			if (img.indexOf("-hover")<0) {
				img = img.replace(/\.png/, "-hover.png")
				$(this).attr("src", img);
			}
			$(this).attr("src", img);
		}, function() {
			var img = $(this).attr("src");
			img = img.replace(/-hover\.png/, ".png")
			$(this).attr("src", img);
		}
	);

// Used in boosted rules
	$('.loadingElement').each(function(idx) {
		var loadingElement = $(this);
		var cellarId = $(this).prev('input[type=hidden]').val();
		$.ajax({
			type: "GET",
			cache: false,
			url: 'search.html',
			data: {'type':'boosted', 'cellarId':cellarId, 'qid':$("[name=qid]").first().val()},
			success: function(data){
				if(data != ''){
					//Replace the ids of boosting rules in order to avoid duplicate ids in search results 
					$("#boostResults").append(data.replace(/\bSearchResult\b/g, "SearchResult BoostedSearchResult")
						.replace(/selectedDocumentColumn[1-9]*/g,'boostedSelectedDocumentColumn'+ idx)
						.replace(/MoreSR_[1-9]*/g, 'boostedMoreSR'+ idx)
					);
					loadingElement.remove();
					// Initiallize the collapsable area of the show more/less which is loaded in the dom after the document-ready has run and it was not initialized along with the other elements.
					initilizeCollapsableElements();
                    //EURLEXNEW-4374: Add thumbtack to boosted results
					$("#boostResults").find('.BoostedSearchResult').prepend("<i class='fa fa-thumb-tack' aria-hidden='true'></i>");
				} else {
					loadingElement.remove();
				}
			}
		});
	});

	$('a.zoomHidden').first().click();

	if($(".dateLine").length > 0)
		dateLineTemplate = ($(".dateLine").first().html());
	$.ajaxSetup({ cache: false });
	// Init Ajax navigation

//
	/*try{
		$.history.init(callbackHistory);
		$("a.history").click(function(){
			$.history.load(this.href.replace(/^.*#/, ''));
			return false;
		});
	}
	catch(e){return;}*/

	//Delete link in Saved rss
	// While click on the link with id deleteRsslink, firstly is checked if any checkbox is selected. 
	// If not eurlexModal css class is removed and a warning is displayed in the same page.
	//If at least one checkbox is selected the eurlexModal class is added and the confirmation modal is displayed.
	$('#deleteRsslink').on('click', function(e){
		if(!isCheckboxSelected("checkbox_")){
			$('#deleteRsslink').removeClass("eurlexModal");
			e.preventDefault();
			$("#warningNoCheckboxSelected").removeClass("hidden");
			$(".alert-success").addClass("hidden");
		}
		else{
			$('#deleteRsslink').addClass("eurlexModal");
			$("#warningNoCheckboxSelected").addClass("hidden");
			$(".alert-success").addClass("hidden");
		}
	});
	//Delete link in Saved items (for documents)
	// While click on the link with id deleteDocumentlink, firstly is checked if any checkbox is selected. 
	// If not eurlexModal css class is removed and a warning is displayed in the same page.
	//If at least one checkbox is selected the eurlexModal class is added and the confirmation modal is displayed.
	$('#deleteDocumentlink').on('click', function(e){
		if(!isCheckboxSelected("selectedDocument")){
			$('#deleteDocumentlink').removeClass("eurlexModal");
			e.preventDefault();
			$(".alert-warning").addClass("hidden");
			$("#warningNoCheckboxSelected").removeClass("hidden");
			$(".alert-success").addClass("hidden");
		}
		else{
			$('#deleteDocumentlink').addClass("eurlexModal");
			$(".alert-warning").addClass("hidden");
			$(".alert-success").addClass("hidden");
		}
	});
	//Delete link in Saved items (for folders)
	// While click on the link with id deleteFolderLink, firstly is checked if any checkbox is selected. 
	// If not eurlexModal css class is removed and a warning is displayed in the same page.
	//If at least one checkbox is selected the eurlexModal class is added and the confirmation modal is displayed.
	$('#deleteFolderLink').on('click', function(e){
		if(!isCheckboxSelected("checkbox_")){
			$('#deleteFolderLink').removeClass("eurlexModal");
			e.preventDefault();
			$("#warningNoCheckboxSelected").removeClass("hidden");
			$(".alert-success").addClass("hidden");
		}
		else{
			$('#deleteFolderLink').addClass("eurlexModal");
			$("#warningNoCheckboxSelected").addClass("hidden");
			$(".alert-success").addClass("hidden");
		}
	});
	//Delete link in Search Preferences
	// While click on the link with id deletePreferencesLink, firstly is checked if any checkbox is selected. 
	// If not eurlexModal css class is removed and a warning is displayed in the same page.
	//If at least one checkbox is selected the eurlexModal class is added and the confirmation modal is displayed.
	$('#deletePreferencesLink').on('click', function(e){
		if(!isCheckboxSelected("checkbox_")){
			$('#deletePreferencesLink').removeClass("eurlexModal");
			e.preventDefault();
			$("#warningNoCheckboxSelected").removeClass("hidden");
			$(".alert-success").addClass("hidden");
		}
		else{
			$('#deletePreferencesLink').addClass("eurlexModal");
			$("#warningNoCheckboxSelected").addClass("hidden");
			$(".alert-success").addClass("hidden");
		}
	});
	//Save items link in Search results
	// While click on the link with id deleteDocumentlink, firstly is checked if any checkbox is selected. 
	// If not eurlexModal css class is removed and a warning is displayed in the same page.
	//If at least one checkbox is selected the eurlexModal class is added and the confirmation modal is displayed.
	$('div#SearchCriteriaPanel #link-save-document').on('click', function(e){
		if(!isCheckboxSelected("selectedDocument")){
			$('div#SearchCriteriaPanel #link-save-document').removeClass("eurlexModal");
			e.preventDefault();
			$(".alert-warning").addClass("hidden");
			$("#warningNoCheckboxSelected").removeClass("hidden");
			$(".alert-success").addClass("hidden");
			$(".alert-info").addClass("hidden");
		}
		else{
			$('div#SearchCriteriaPanel #link-save-document').addClass("eurlexModal");
			$(".alert-success").addClass("hidden");
			$(".alert-info").addClass("hidden");
			$(".alert-warning").addClass("hidden");
		}
	});
	//Clear selection link in Search results
	// While click on the link with id start with clearSelectedCheckboxes_, firstly is checked if any checkbox is selected. 
	// If not eurlexModal css class is removed and a warning is displayed in the same page.
	//If at least one checkbox is selected the eurlexModal class is added and the confirmation modal is displayed.
	$("a[id^='clearSelectedCheckboxes_']").on('click', function(e){
		if(!isCheckboxSelected("selectedDocument")){
			$("a[id^='clearSelectedCheckboxes_']").removeClass("eurlexModal");
			e.preventDefault();
			$(".alert-warning").addClass("hidden");
			$("#warningNoCheckboxSelected").removeClass("hidden");
			$(".alert-success").addClass("hidden");
			$(".alert-info").addClass("hidden");
			$('html, body').animate({ scrollTop: 0 }, 'fast');
		}
		else{
			$("a[id^='clearSelectedCheckboxes_']").addClass("eurlexModal");
			$(".alert-success").addClass("hidden");
			$(".alert-info").addClass("hidden");
			$(".alert-warning").addClass("hidden");
		}
	});

	highlightExpertQuery();

	removeWebtrendsCookies();
}// End of document ready


function GetIEVersion() {
	var sAgent = window.navigator.userAgent;
	var Idx = sAgent.indexOf("MSIE");

	// If IE, return version number.
	if (Idx > 0)
		return parseInt(sAgent.substring(Idx+ 5, sAgent.indexOf(".", Idx)));

	// If IE 11 then look for Updated user agent string.
	else if (!!navigator.userAgent.match(/Trident\/7\./))
		return 11;

	else
		return 0; //It is not IE
}


// Fix IE bloking for very large modal, by following a similar to no-js approach where the modal opens in a new page
function flatlistHeavyModalHandling(subdomain) {
	// In case we have internet explorer we dont show modal but we open a separate page.
	if (GetIEVersion() > 0) {
		$('#authorSubmitBtn').click()
	}
	else {
		var url = "flat-list.html?fromId=authorOther&fillType=fillHierarchyForm&subdomain=" + subdomain + "&code=AU_CODED";
		showModal(url, 'fillHierarchyTree()', 'AU,generatedHierarchyValues_Author');
	}
}


//return object where class is "modal onlyJsInline"
function saveHrefExport(){
	var element= $("a#link-export-documentTop.exportSelection, a#link-export-documentBottom.exportSelection");
	if (typeof(element)!='undefined') return element;
	else return null;
}
// DROPDOWN
var myTimeout;

function resetSubInSec(){
	myTimeout = window.setTimeout(resetSub,200);
}

function cancelTimeOut(){
	clearTimeout(myTimeout);
}

function showSubMenu(el){
	cancelTimeOut();
	resetSub();
	var leftpx=$(el).position().left +'px';
	var minSize=Math.max($(el).width(), 200);
	var liList = $(el).parent('li').parent('ul').children('li');
	for (var i = 0, len = liList.length; i < len; i++) {
		minSize=Math.max($(liList[i]).children('a').width()+18, minSize);
	}
	var minSizepx=minSize +'px';

	// var toppx=$(el).position().top + $(el).innerHeight() +'px';
	var heigthTopMenu=$(el).parent('li').parent('ul').height();
	var paddingTopMenu=8;
	// var heigthTopMenu=22;
	var toppx=paddingTopMenu + heigthTopMenu +'px';

	$(el).addClass('lv1Hover');
	$(el).parent('li').children('ul').css({left:leftpx,top:toppx,display:'block'});
	$(el).parent('li').children('ul').css("width",minSizepx);
	$(el).parent('li').children('ul').children('li').css("width",minSizepx);
	$(el).parent('li').children('ul').animate({opacity:'1'},200);
}
function resetSub(){
	$('.lv1Hover').removeClass('lv1Hover')
	$('.subM').css({left:'545em',display:'none',opacity:'0.2'});
}
// END DROPDOWN


function checkOpBannerDisplay(){
	var larg2el = $('.op-quicksearch').innerWidth();
	if( $('#op-header').innerWidth() > 470 + larg2el ){
		$('#op-header').addClass('forNormalScreen');
	}
	else{
		$('#op-header').removeClass('forNormalScreen');
	}
}

/*function callbackHistory(hash)
{
	if(hash.indexOf('nal-') < 0){
		$("#documentView a.selectedTab").removeClass("selectedTab");
		if(hash){
			$(".tabContent").load(hash+".html .forAjaxLoading", ajaxDocumentReady );
			$("."+hash+" a.history").addClass("selectedTab");
		}
		else{
			var alternativeUrl = (location.href).substring(location.href.lastIndexOf("/")+1);
			var UrlWithoutExt = alternativeUrl.substring(0,alternativeUrl.lastIndexOf("."));
			$(".tabContent").load(alternativeUrl+" .forAjaxLoading", ajaxDocumentReady );
			$("."+UrlWithoutExt+" a.history").addClass("selectedTab");
		}
	}
}*/

function ajaxDocumentReady(){
	$(".hideInJs").css({display:'none'});
	$(".absoluteInJs").css({position:'absolute', left:'-9995em'});
	$(".jsOnly").css({display:'block'});
}
/* Callback */
/*
 * var dynParam = new Array(); dynParam.push('34'); callFunction('testAlert',
 * dynParam);
 */
function callFunction(fnc, prm){
	if (typeof(fnc)!='undefined' && '' != fnc) {
		if(typeof(prm)!='undefined' && '' != prm) {
			eval(fnc)(prm);
		} else {
			eval(fnc);
		}
	}
}
/*
 * function testAlert(prm){ alert(prm[0]); }
 */
/* Fin callBack */

function callbackMetadataTree() {
	loadMetadataCheckBoxes();
}
function callbackMetadataTreeAjaxDocReady() {
	$('#zoomSelectionLi').removeClass('hide');
	$('#advancedSelectionLi').removeClass('active');
	$('#simpleSelectionLi').removeClass('hide').addClass('active');
	$('#tabs-5').removeClass('active').hide();
	$('#tabs-4').addClass('active').show();
	initMetadataFilters();
	loadMetadataCheckBoxes();
}

function updateMetadataFilterOnEnter(e){
	var isAdvancedSelected;
	var simpleTab = $("#tabs-4");
	var advancedTab = $("#tabs-5");
	var simpleTabInputField = $("#metadataFilterSimple");
	var advancedTabInputField = $("#metadataFilterAdvanced");
	var simpleTabFilterButton = $('input#metadataFilterSimpleButton');
	var advancedTabFilterButton = $('input#metadataFilterAdvancedButton');

	if (simpleTab != 0 && simpleTab.hasClass('active')){
		isAdvancedSelected = false;
	}else if (advancedTab !=0 && advancedTab.hasClass('active')){
		isAdvancedSelected = true;
	}else{
		return false;
	}
	e = e || window.event;
	var code = e.keyCode ? e.keyCode : e.which;
	//if ENTER button pressed clicks on Filter button in "Change displayed metadata" section
	if (code === 13){
		e.preventDefault();
		if (isAdvancedSelected){
			advancedTabFilterButton.click();
			advancedTabInputField.click();
		} else{
			simpleTabFilterButton.click();
			simpleTabInputField.click();
		}
	}
}

function initMetadataFilters() {
	$('#metadataFilterSimpleButton').click(function() {
		filterList($('.tab-pane.active .overflowedAllTabs ul'), $('#metadataFilterSimple').val(), 1);
		expandAllTree('metadataGroupTree');
		return false;
	});
	$('#metadataFilterSimple-clearAdvancedSearchIconButton').click(function() {
		filterList($('.tab-pane.active .overflowedAllTabs ul'), $('#metadataFilterSimple').val(), 1);
		return false;
	});
	$('#metadataFilterSimple-clearAdvancedSearchButton').click(function() {
		return false;
	});
	$("#metadataFilterAdvancedButton").click(function() {
		filterList($('.tab-pane.active .overflowedAllTabs ul'), $('#metadataFilterAdvanced').val(), 1);
		expandAllTree('metadataGroupTree');
		return false;
	});
	$('#metadataFilterAdvanced-clearAdvancedSearchIconButton').click(function() {
		filterList($('.tab-pane.active .overflowedAllTabs ul'), $('#metadataFilterAdvanced').val(), 1);
		collapseAdvancedEmptyMetadataGroup();
		return false;
	});
	$('#metadataFilterAdvanced-clearAdvancedSearchButton').click(function() {
		return false;
	});
}

/*
 * Warning message for "export selection" if no checkbox is checked
 */
//returns true if at least one checkbox is selected of "docCheckbox onlyJs" class else false
function isExportSelected(){
	var checkboxes = $(".docCheckbox.onlyJs");
	for (var i=0;i<checkboxes.length;i++){
		if (checkboxes[i].checked && checkboxes[i].checked !=null ) return true;
	}
	return false;
}
//checks if isExportSelected and removes or adds eurlexModal attribute to link accordingly
function checkAndExportSelection(e){
	if (!isExportSelected()){
		$('#link-export-documentTop, #link-export-documentBottom').removeClass("eurlexModal");
		e.preventDefault();
		$(".alert-warning").addClass("hidden");
		$("#warningExp").removeClass("hidden");
		$(".alert-success").addClass("hidden");
		$(".alert-info").addClass("hidden");
		$('html, body').animate({ scrollTop: 0 }, 'fast');
	}
	else{
		$('#link-export-documentTop, #link-export-documentBottom').addClass("eurlexModal");
		$(".alert-warning").addClass("hidden");
		$(".alert-success").addClass("hidden");
		$(".alert-info").addClass("hidden");
	}
}



/*
 * ==================== COOKIES =============================
 */

/**
 *  Instantly toggles the state of expandable content.
 *
 *  @param  element  The element whose 'expanded' state
 *                   is to be toggled.
 */
function smartToggle(element) {
    if (typeof element === 'string') {
        // Apply HTML sanitization for security reasons.
        element = sanitizeHtml(element);
    }

    element = $(element);

    // Add the 'notransition' CSS class temporarily to
    // skip activating a transition animation.
    element.addClass('notransition').collapse('toggle');

    const transitionTimeoutID = setTimeout(function() {
        // Restore normal transition behavior.
        element.removeClass('notransition');
    }, 200);
    element.data('timeout', transitionTimeoutID);
}

//toggle the trees in change displayed metadata
function smartToogleTree(el) {
	var isVisible = $(el).css('display') !== 'none';
	var nav = navigator.userAgent.toLowerCase();

	// EURLEXNEW-3970: ECB-Statistics widget should adjust its height to that of ECB-Tree
	// upon expanding/collapsing of nodes and completion of transition effects.
	var callback;
	if ($("#EcbMenuBlock1").length){
		callback = function() {$("#EcbMenuBlock1").trigger("treeResize")};
	}
	else {
		callback = function() {};
	}

	if (nav.indexOf("msie 6") > -1 || nav.indexOf("msie 7") > -1) {
		// $(el).toggle(); unexpected behavior:
		// http://bugs.jquery.com/ticket/11436
		if (isVisible) {
			$(el).hide(callback);
		} else {
			$(el).show(callback);
		}
	} else {
		// $(el).slideToggle(); unexpected behavior:
		// http://bugs.jquery.com/ticket/11436
		if (isVisible) {
			$(el).slideUp().fadeOut(callback);
		} else {
			$(el).slideDown().show(callback);
		}
	}
}

/**
 *  Toggles the 'expanded' state of metis menu expandable content
 *  that is controlled by the provided metis menu 'trigger' element.
 *
 *  @param  trigger  The 'trigger' element/selector of the metis menu.
 */
function searchResultsMetisMenuToggle(trigger) {
    // Apply HTML sanitization to the 'trigger' parameter for security
    // reasons.
    if (typeof trigger === 'string') trigger = sanitizeHtml(trigger);

    trigger = $(trigger);

    getCookieFromDoc().split(';').forEach(function(cookie) {
        // Get the name & value for the current cookie, and also apply
        // HTML sanitization for security reasons.
        const cookieName = sanitizeHtml(cookie.split('=')[0]);
        const cookieValue = sanitizeHtml(readCookie(cookieName));

        const triggerParent = trigger.parents('.CollapseTreeMenu-sm');
        const content = trigger.siblings();

        // Find the 'container' element for the current cookie, and
        // check if the 'expanded' state of the corresponding content
        // should be toggled.
        const container = $('#' + cookieName.trim()).parents();
        const isContentExpanded = container.hasClass('Expanded');
        if (isContentExpanded && (cookieValue === '0')) {
            content.removeClass('in').attr('aria-expanded', 'false');
            triggerParent.removeClass('Expanded');
            trigger.attr('aria-expanded', 'false');
        } else if (!isContentExpanded && (cookieValue === '1')) {
            content.addClass('in').attr('aria-expanded', 'true');
            triggerParent.addClass('Expanded');
            trigger.attr('aria-expanded', 'true');
        }
    });
}

function createCookie(name,val,expDays) {
	// Check if the user accept to store cookies, and if he did it, store the
	// cookie.
	if($wt.cookie.exists('cck1')) {
		var cck1 = JSON.parse($wt.cookie.get("cck1"));
	}
	if(cck1.all1st || cck1.cm){
		if (expDays > maxNbDayForCookie) {
			expDays = maxNbDayForCookie;
		}
		var expireDate = new Date();
		expireDate.setTime(expireDate.getTime() + expDays*24*3600*1000);
		document.cookie = name + "=" + escape(val) + "; expires=" + expireDate.toGMTString() +"; path=/;";
	}
	// Else just keep the cookie in session mode (expire in 0)
	else {
		document.cookie = name + "=" + escape(val) + "; path=/;";
	}
}

function readCookie(name) {
	var st,end;
	// checks existence
	st = getCookieFromDoc().indexOf(name + "=");
	// It found a cookie with the specified name
	if (st >= 0) {
		// Compute where the name ends. Position of =
		st += name.length + 1;
		// find the first occurance of the ";" after the st string
		end = getCookieFromDoc().indexOf(";",st);
		// Does not exist ";" - end of cookie FULL string  
		if (end < 0) {
			end = getCookieFromDoc().length;
		}
		//the value of the cookie name after = 
		return unescape(getCookieFromDoc().substring(st,end));
	}
	return "";
}


function deleteCookie(name) {
	var cookieToExpire = name + "=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
	document.cookie = cookieToExpire;
	// If cookie still exists, retry using domain as well.
	if (readCookie(name) != "") {
		document.cookie = cookieToExpire + " domain=" + window.location.hostname + ";";
	}
}

function getCookie(cookie) {
	var name;
	var value;
	var cookies=getCookieFromDoc().split(";");
	for(var i=0;i<cookies.length;i++) {
		name=cookies[i].substr(0,cookies[i].indexOf("="));
		value=cookies[i].substr(cookies[i].indexOf("=")+1);
		name=name.replace(/^\s+|\s+$/g,"");
		if (name==cookie){
			return unescape(value);
		}
	}
}




// Detects if enter is pressed in advanced search form so as to decide whether to submit
function advancedSearchFormFreeTextKeyPressed(myfield,e) {
	var keycode;
	if (window.event) keycode = window.event.keyCode;
	else if (e) keycode = e.which;
	else return true;

	if (keycode == 13) {
		if($('.focus').length > 0){
			$(myfield).val($('.focus').text());
		}
		return false;
	}
}

function disableFormSubmit(myfield) {
// myfield.form.setAttribute("onsubmit", "return false;");
	formSubmit = false;
}

function enableFormSubmit(myfield) {
// myfield.form.setAttribute("onsubmit", "return true;");
	formSubmit = true;
}

function toggle(id) {
	$("#"+id).toggle();
}


/**** Document notice panels -  Expand collapse depending on cookie stored value *****/
function createDocPartCookie(el){
	if( $(el).hasClass('collapsed') ){
		createCookie($(el).closest('.panel-heading').attr('id'),'1',30);
	}else{
		createCookie($(el).closest('.panel-heading').attr('id'),'0',30);
	}
}

function createDocPartCookieSearch(el){
	if ($(el).parents().hasClass('Expanded')){
		createCookie($(el).attr("id"),'0',30);
	}else{
		createCookie($(el).attr("id"),'1',30);
	}
}

//Collapse panel on-load in mobile devices (start)
var initilizeCollapsableElements = function() {
	if (window.innerWidth < 992) {
		$('.CollapsePanel-sm .collapse').removeClass('in');
		$('.CollapsePanel-sm .panel-title a').addClass('collapsed');
		$('.CollapsePanel-sm .panel-title a').attr('aria-expanded', 'false');
	} else {
		$('.CollapsePanel-sm .collapse').addClass('in');
		$('.CollapsePanel-sm .panel-title a').removeClass('collapsed');
		$('.CollapsePanel-sm .panel-title a').attr('aria-expanded', 'true');
	}

	if (window.innerWidth < 768) {
		$('.CollapsePanel-xs .collapse').removeClass('in');
		$('.CollapsePanel-xs .panel-title a').addClass('collapsed');
		$('.CollapsePanel-xs .panel-title a').attr('aria-expanded', 'false');
	} else {
		$('.CollapsePanel-xs .collapse').addClass('in');
		$('.CollapsePanel-xs .panel-title a').removeClass('collapsed');
		$('.CollapsePanel-xs .panel-title a').attr('aria-expanded', 'true');
	}
}

var initializeMutuallyExclusizeCollapsablePanels = function() {
	//If we have mutually exclusive panels
	if ($('.mutuallyExclusivePanels').length > 0){
		var hasErrorPanel = $('.mutuallyExclusivePanels .has-error').length > 0;
		$('.mutuallyExclusivePanels .panel').each(function( index ) {
			//If we have panel inside with validation errors
			if ( $(this).find('.has-error').length > 0 ){
				//expand them
				$(this).find('.collapse').addClass('in');
				$(this).find('.panel-title a').removeClass('collapsed');
				$(this).find('.panel-title a').attr('aria-expanded', 'true');
				$(this).find('.panel-title button').removeClass('collapsed');
                $(this).find('.panel-title button').attr('aria-expanded', 'true');
				//and close its siblings
				closeSiblingPanels($(this));
			} else { //do what is defined by cookies if anything, unless we have some expanded errorPanel
				if ( $(this).hasClass('expandByCookie') && !hasErrorPanel ){
					//expand defined by cookies
					$(this).find('.collapse').addClass('in');
					$(this).find('.panel-title a').removeClass('collapsed');
					$(this).find('.panel-title a').attr('aria-expanded', 'true');
					$(this).find('.panel-title button').removeClass('collapsed');
                    $(this).find('.panel-title button').attr('aria-expanded', 'true');
					//close sibling panels
					closeSiblingPanels($(this));
				} else if ( $(this).hasClass('collapseByCookie') && !hasErrorPanel ){
					//collapse defined by cookies
					$(this).find('.collapse').removeClass('in');
					$(this).find('.panel-title a').addClass('collapsed');
					$(this).find('.panel-title a').attr('aria-expanded', 'false');
					$(this).find('.panel-title button').addClass('collapsed');
                    $(this).find('.panel-title button').attr('aria-expanded', 'false');
				}
			}
		});
	}
}

function closeSiblingPanels(panel){
	panel.siblings().each(function( index ) {
		$(this).find('.collapse').removeClass('in');
		$(this).find('.panel-title a').addClass('collapsed');
		$(this).find('.panel-title a').attr('aria-expanded', 'false');
		$(this).find('.panel-title button').addClass('collapsed');
        $(this).find('.panel-title button').attr('aria-expanded', 'false');
	});
}
// Collapse panel on-load in mobile devices (end)

/**
 *  Toggles the 'expanded' state of the 'Search criteria' panel
 *  in the 'Search results' page, based on saved user preferences.
 *
 *  @param  elementClass  The CSS class to use as an ID, to
 *                        isolate all the relevant cookies that
 *                        hold the required information.
 */
function loadDocStateByClass(elementClass) {
    // Apply the default state of the 'Search criteria' panel, in
    // the search results page.
    applySearchResultsDefaults();

    // Check if there are any cookies that may hold relevant
    // information.
    if (getCookieFromDoc().indexOf(elementClass) === -1) return;

    getCookieFromDoc().split(';').forEach(function(cookie) {
        // Get the name & value for the current cookie, and also
        // apply HTML sanitization for security reasons.
        const cookieName = sanitizeHtml(cookie.split('=')[0]);
        const cookieValue = sanitizeHtml(readCookie(cookieName));

        // Toggle the 'expanded' state of the 'Search criteria'
        // panel as required.
        const trigger = $('#' + cookieName.trim()).find('a');
        const isContentCollapsed = trigger.hasClass('collapsed');
        if ((isContentCollapsed && (cookieValue === '1')) ||
            (!isContentCollapsed && (cookieValue === '0'))) {
            const elementID = '#' + cookieName.replace(/ /g, '');

            toggleNextElement('.' + elementClass + ' ' + elementID);
            toggleNextElement(elementID);
        }
    });
}

//default state of search criteria in search results page
function applySearchResultsDefaults(){
	if(window.innerWidth < 992){
		//set search criteria to be collapsed
		$("#SearchCriteriaPanel").removeClass('in');
		$("#SearchCriteriaPanelTitle .panel-title button" ).addClass('collapsed');
		$("#SearchCriteriaPanelTitle .panel-title button").attr('aria-expanded', 'false');
	}

}

/**
 *  Toggles the 'expanded' state of a tab's expandable content for a notice
 *  document's view, based on saved user preferences.
 *
 *  @param  tabID     The tab ID of a notice document's view, on which to
 *                    apply user preferences regarding expandable content.
 *  @param  cookieID  The ID to use to isolate all the relevant cookies that
 *                    hold the required information.
 */
function loadDocState(tabID, cookieID) {
    applyNoticeDefaults(tabID); // Apply the default state for the tab.

    // Check if there are any cookies that may hold relevant information.
    if (getCookieFromDoc().indexOf(cookieID) === -1) return;

    getCookieFromDoc().split(';').forEach(function(cookie) {
        // Get the name & value for the current cookie, and also apply HTML
        // sanitization for security reasons.
        let cookieName = sanitizeHtml(cookie.split('=')[0]);
        const cookieValue = sanitizeHtml(readCookie(cookieName));

        // Find the trigger, and check if its controlled content is eligible
        // for 'expanded' state toggling.
        let trigger = $('#' + cookieName.trim()).find('a');
        if (trigger.hasClass('blockedByUserPreference')) return;

        // Toggle the 'expanded' state of the controlled content as required.
        const isContentCollapsed = trigger.hasClass('collapsed');
        if ((isContentCollapsed && (cookieValue === '1')) ||
            (!isContentCollapsed && (cookieValue === '0'))) {
            toggleNextElement('#' + cookieName.replace(/ /g, ''));
        }

        /* EURLEXNEW-3644: 'Procedure' view logic. */
        if ((cookieID === 'Procedure') && (cookieName.indexOf('Procedure') > -1)) {
            cookieName = cookieName.replace(cookieID, '').trim();

            const panel = $('#' + cookieName);
            if (!panel.hasClass('in') && (cookieValue === '1')) {
                panel.collapse('show'); // Expand the controlled content.

                // For the current element, also update the trigger's 'plus'/'minus'
                // icon accordingly.
                trigger = $(".ViewMoreInfo[aria-controls='" + cookieName + "']");
                trigger.find('i.fa').toggleClass(["fa-plus-square", "fa-minus-square"]);
            }
        }
    });
}

//Default state of notice page
function applyNoticeDefaults(stateOfTabs) {
	if (window.innerWidth < 992) {
		// Check if the selected tab of notice page is Text  and not Text tab of Summaries of EU Legislation (==false).
		if (stateOfTabs == "TXT false") {
			$(".panel-heading").map(
				function(index, obj) {
					var currentIterationID = obj.id;
					var elementContents = "#"+ $(this).siblings().attr('id');
					var elementContentsPanel = "#"+ currentIterationID.concat(' .panel-title button');
					// Set each element to be collapsed apart from "Title and reference" and "Text" sections
					$(elementContents).not("#PP1Contents, #PP4Contents, #PP4ContentsPdf").removeClass('in');
					$(elementContentsPanel).not("#PP1 .panel-title button, #PP4 .panel-title button,#PP4Pdf .panel-title button").addClass('collapsed');
					$(elementContentsPanel).not("#PP1 .panel-title button, #PP4 .panel-title button,#PP4Pdf .panel-title button").attr('aria-expanded', 'false');
				});
			// Check if the selected tab of notice page is Summary of legislation or Summary or Text tab of Summaries of EU Legislation (==true)
		} else if((stateOfTabs == "LSU") || (stateOfTabs == "SUM") || (stateOfTabs == "TXT true") ) {
			$(".panel-heading").map(
				function(index, obj) {
					var currentIterationID = obj.id;
					var elementContents = "#"+ $(this).siblings().attr('id');
					var elementContentsPanel = "#"+ currentIterationID.concat(' .panel-title button');
					// Set each element to be collapsed apart from "Text" section
					$(elementContents).not("#PP4Contents, #PP4ContentsPdf").removeClass('in');
					$(elementContentsPanel).not("#PP4 .panel-title button, #PP4Pdf .panel-title button").addClass('collapsed');
					$(elementContentsPanel).not("#PP4 .panel-title button, #PP4Pdf .panel-title button").attr('aria-expanded', 'false');
				});
			// Check if the selected tab of notice page is Document information or National transposition or Draft national legislation or Technical working document
		} else if ((stateOfTabs == "ALL") || (stateOfTabs == "DNL") || (stateOfTabs == "TWD") || (stateOfTabs == "NIM")) {
			$(".panel-heading").map(
				function(index, obj) {
					var currentIterationID = obj.id;
					var elementContents = "#"+ $(this).siblings().attr('id');
					var elementContentsPanel = "#"+ currentIterationID.concat(' .panel-title button');
					// Set the "Multilingual display" section to be collapsed. All other sections will be expanded.
					$("#PP3Contents").removeClass('in');
					$("#PP3 .panel-title button").addClass('collapsed');
					$("#PP3 .panel-title button").attr('aria-expanded', 'false');
				});
			// Check if the selected tab of notice page is Procedure or Internal procedure
		} else if((stateOfTabs == "PROC") || (stateOfTabs == "HIS")|| (stateOfTabs == "PIN")) {
			$(".panel-heading").map(
				function(index, obj) {
					var currentIterationID = obj.id;
					var elementContents = "#"+ $(this).siblings().attr('id');
					var elementContentsPanel = "#"+ currentIterationID.concat(' .panel-title button');
					// Set each element to be collapsed
					$(elementContents).removeClass('in');
					$(elementContentsPanel).addClass('collapsed');
					$(elementContentsPanel).attr('aria-expanded', 'false');
				});
		}
	}
}

// Cookies for mutually exclusive panels - widgets
function createDocPartCookieWidgets(el){
	if( $(el).hasClass('collapsed')) {
		$(".panel-heading").map(function(index, obj) {
			var currentIterationID = obj.id
			createCookie(currentIterationID, '0', 30);
			createCookie($(el).closest('.panel-heading').attr('id'),'1',30);
		});
	}else{
		$(".panel-heading").map(function(index, obj) {
			var currentIterationID = obj.id
			createCookie(currentIterationID, '0', 30);
		});
	}
}
// Widgets expand - collapse depending on cookie storage
function loadDocWidgets(elClass) {
	applyWidgetsDefaults();
	var cookieNameString = getCookieFromDoc();
	// Check if the name of the state exists in the cookie string
	var cookieWithState = cookieNameString.indexOf(elClass);
	if (cookieWithState != '-1') {
		for (var i = 0; i < getCookieFromDoc().split(';').length; i++) {
			var cooVal = readCookie((getCookieFromDoc().split(';')[i]).split("=")[0]);
			var cookieNameString = sanitizeHtml("#" + (getCookieFromDoc().split(';')[i].split("=")[0].trim()));
			if (($(cookieNameString).parent().hasClass('expandByCookie') && (cooVal == '0'))){
				$(cookieNameString).parent().removeClass("expandByCookie");
			}else if((!$(cookieNameString).parent().hasClass('expandByCookie') && (cooVal == '1'))){
				$(cookieNameString).parent().addClass("expandByCookie");
			}
		}
	}
}

//Default state of widgets
//NOTE: It depends on two document ready funcs
function applyWidgetsDefaults() {
	//If we have mutually exclusive panels
	var elementExpanded = $("#WDGPanels .panel-heading").attr('id');
	if ($('.mutuallyExclusivePanels').length > 0) {
		$("#" + elementExpanded).parent().addClass("expandByCookie");
	}
	// In case of mobile view
	if (window.innerWidth < 992) {
		if ($('.mutuallyExclusivePanels').length > 0) {
			$("#" + elementExpanded).parent().removeClass("expandByCookie");
		}
	}
}

/**
 *  Locates the 'content' element for a 'trigger' element, and toggles
 *  its 'expanded' state.
 *
 *  @param  trigger  The 'trigger' element/selector.
 */
function toggleNextElement(trigger) {
    if (typeof trigger === 'string') trigger = sanitizeHtml(trigger);

    const content = $(trigger).siblings().attr('id');
    smartToggle($('#' + content));
}

/**
 *  Toggles the state of expandable content, based on saved user preferences.
 */
function loadPreferencedState() {
    getCookieFromDoc().split(';').forEach(function(cookie) {
        // Get the name & value for the current cookie, and also apply HTML
        // sanitization for security reasons.
        const cookieName = sanitizeHtml(cookie.split('=')[0]);
        const cookieValue = sanitizeHtml(readCookie(cookieName));

        // Toggle the 'expanded' state of the controlled content as required.
        const trigger = $('#' + cookieName.trim()).find('button');
        const isContentCollapsed = trigger.hasClass('collapsed');
        if ((isContentCollapsed && (cookieValue === '1')) ||
            (!isContentCollapsed && (cookieValue === '0'))) {
            smartToggle($('#' + cookieName.replace(/ /g, '') + 'Contents'));
        }
    });
}

/* Legal content tab - cookies */
function collapseAll() {
	$(".PagePanel .panel-heading").map(function(index, obj) {
		var currentIterationID = obj.id
		createCookie(currentIterationID, '0', 30);
	});

}
function expandAll() {
	$(".PagePanel .panel-heading").map(function(index, obj) {
		var currentIterationID = obj.id
		createCookie(currentIterationID, '1', 30);
	});
}

/**
 *  Toggles the 'expanded' state of the metis menu in the search
 *  results page, based on saved user preferences.
 *
 *  @param  cookieID  The ID to use to isolate all the relevant
 *                    cookies that hold the required information.
 */
function loadPreferencedStateSearchResults(cookieID) {
    applyMetisMenuDefaults(); // Apply metis menu default state.

    // Check if there are any cookies that may hold relevant
    // information.
    if (getCookieFromDoc().indexOf(cookieID) === -1) return;

    getCookieFromDoc().split(';').forEach(function(cookie) {
        // Get the name & value for the current cookie, and also
        // apply HTML sanitization for security reasons.
        const cookieName = sanitizeHtml(cookie.split('=')[0]);
        const cookieValue = sanitizeHtml(readCookie(cookieName));

        // Find the 'container' element for the current cookie,
        // and check if its content is eligible for 'expanded'
        // state toggling.
        const container = $('#' + cookieName.trim()).parents();
        const isContentExpanded = container.hasClass('Expanded');
        if ((isContentExpanded && (cookieValue === '0')) ||
            (!isContentExpanded && (cookieValue === '1'))) {
            // Find the corresponding metis menu 'trigger' element.
            const trigger = $('#' + cookieName.replace(/ /g, ''));

            if (trigger.hasClass('widgetControl')) {
                searchResultsMetisMenuToggle(trigger);
            }
        }
    });
}

// Default state of metis menu in search results page
function applyMetisMenuDefaults(){
	if(window.innerWidth < 992){
		$('.CollapseTreeMenu-sm').removeClass('Expanded');
		$('.CollapseTreeMenu-sm a[aria-expanded="true"]').attr('aria-expanded', 'false');
		$('.CollapseTreeMenu-sm ul').removeClass('in').attr('aria-expanded', 'false');
	}else{
		$('.CollapseTreeMenu-sm').addClass('Expanded');
		$('.CollapseTreeMenu-sm a[aria-expanded="false"]').attr('aria-expanded', 'true');
		$('.CollapseTreeMenu-sm ul').addClass('in').attr('aria-expanded', 'true');
	}
}

/* Widget expand/collapse border line */
$(document).on('mouseup keydown keyup', function(e){
	//contact-attach border line
	if ($("#attachFileId").has(e.target).length === 0){
		$("#attachFileId").removeClass("borderFocus");
	}
	//my search-browse border line
	if ($("#browseId").has(e.target).length === 0){
		$("#browseId").removeClass("borderFocus");
	}
});

function getCookieFromDoc() {
    let cookie = document.cookie;
    return cookie != null ? cookie : '';
}

/**** Advanced search panels collapse expand depending on cookie values *****/
/* Advanced search form section expand/collapse */
function loadAdvancedPreferencedState(advancedFormState, allAdvancedState) {
	// Applies the default state
	applyAdvancedDefaults();
	var cookieNameString = getCookieFromDoc();
	//check if the name of the collection state exists in the cookie string 
	var cookieWithState = cookieNameString.indexOf(advancedFormState);
	var genCooVal = readCookie(allAdvancedState);
	var formCooVal = getCookie(advancedFormState);
	// Case that cookie for this page exists and we are in mobile: Apply the cookies state
	// Case Cookie for this page exist and Desktop: Apply cookies.
	// Nothing is done for cases that: Cookie does not exist.  and we are in mobile AND cookie does not exist and Desktop 
	if (genCooVal == '0' && (formCooVal == null || formCooVal != genCooVal)) {
		collapseAllAdvancedState(advancedFormState, allAdvancedState);
		loadPreferencedState();
	}else if(genCooVal == '1' && (formCooVal == null || formCooVal != genCooVal)) {
		expandAllAdvancedState(advancedFormState, allAdvancedState);
		loadPreferencedState();
	}else if (cookieWithState != '-1') {
		loadPreferencedState();
	}
}

function collapseAllAdvancedState(id,allAdvancedState){
	createCookie(allAdvancedState,'0',30);
	createCookie(id,'0', 30);
	$(".col-md-3 .panel-heading").map(function(index, obj) {
		var currentIterationID = obj.id
		createCookie(currentIterationID,'0',30);
	});

}
function expandAllAdvancedState(id,allAdvancedState){
	createCookie(allAdvancedState,'1',30);
	createCookie(id,'1',30);
	$(".col-md-3 .panel-heading").map(function(index, obj) {
		var currentIterationID =  obj.id
		createCookie(currentIterationID,'1',30);
	});
}


function applyAdvancedDefaults(){
	if(window.innerWidth < 992){
		$(".panel-heading").map(function(index, obj) {
			var currentIterationID = obj.id;
			var elementContents ="#"+ currentIterationID.concat(' .collapse');
			var elementContentsPanel ="#"+ currentIterationID.concat(' .panel-title button');
			var expandedElementID = $(".AdvancedSearchPanel").attr('id');
			var expandedElement ="#"+ expandedElementID.concat(' .collapse');
			var expandedElementPanel ="#"+ expandedElementID.concat(' .panel-title button');
			if (currentIterationID!=expandedElementID){
				//set each element to be collapsed
				$(elementContents).removeClass('in');
				$(elementContentsPanel).addClass('collapsed');
				$(elementContentsPanel).attr('aria-expanded', 'false');
			}else {
				//set the first element to be expanded
				$(expandedElement).addClass('in');
				$(expandedElementPanel).removeClass('collapsed');
				$(expandedElementPanel).attr('aria-expanded', 'true');
			}

		});
	}else{
		$('.AdvancedSearchPanel .collapse').addClass('in');
		$('.AdvancedSearchPanel .panel-title a').removeClass('collapsed');
		$('.AdvancedSearchPanel .panel-title a').attr('aria-expanded', 'true');
		$('.AdvancedSearchPanel .panel-title button').removeClass('collapsed');
        $('.AdvancedSearchPanel .panel-title button').attr('aria-expanded', 'true');
	}

}
/********* Trees - Expand  collapse ************/
function collapseAllTree(treeClass){
	$("."+treeClass+" li" ).each( function(){
		if($(this).children('a').first().children('img').first().length > 0 && $(this).children('a').first().children('img').first().attr('src').indexOf('collapse-tree') > -1)
			$(this).children('a').first().click();
	} );
}

function expandAllTree(treeClass){
	$("."+treeClass+" li" ).each( function(){
		if($(this).children('a').first().children('img').first().length > 0 && $(this).children('a').first().children('img').first().attr('src').indexOf('expand-tree') > -1)
			$(this).children('a').first().click();
	} );
}


// ==========END COOKIES ===========






/************************** MODALS  ******************/

/*Show modal - related functionalities */
// Performs preparation/checks before loading the modal
function showModal(url, callback, fillUrlCode){
	lastFocusElement = $(':focus');
	// Show the spinner and append an empty tempdata container div to the body so as to be used for loading the ajax response
	showHourglass();

	// Creates the full url
	if(typeof(fillUrlCode) != 'undefined' && fillUrlCode != '') {
		url = url + fillUrl(fillUrlCode);
	}
	// Checks if the session has expired and the modal was called from a protected url. If yes it redirects to authentication
	if (sessionExpired && url.match('(.*)\/protected\/(.*)')) {
		var towardUrl = url;
		if (towardUrl.indexOf('/protected') > -1) {
			towardUrl = towardUrl.substring(towardUrl.indexOf('/protected'), towardUrl.length);
		}
		url = authenticationRequiredUrl + '?towardUrl=' + encodeURIComponent(towardUrl) + '&callingUrl=' + encodeURIComponent(pageUrl + '?' + queryString);
	}

	// In case the isAjaxRequest=true is missing it appends it in the URL so as the server to understand that this is an AJAX call
	if(url.indexOf('isAjaxRequest=true') == -1) {
		url = url + (url.indexOf('?') == -1 ? '?' : '&') + 'isAjaxRequest=true';
	}
	// The actual modal loading.
	loadModal(url, callback);
}

function showAuthenticationRequiredModal(url, callback){
	lastFocusElement = $(':focus');

	$("#selectBox").remove();
	// hide <select> useful for ie6
	if(navigator.userAgent.toLowerCase().indexOf("msie 6") > -1){
		var selects = document.getElementsByTagName("select");
		for (var i = 0; i < selects.length; i++) {
			selects[i].style.visibility ="hidden";
		}
	}
	showHourglass();

	//Remove the frontOfficeUrl part (e.g. https://eur-lex.eurodyn.com/) from the full url and keep only the relative Url.
	var towardUrl = url;
	if (towardUrl.indexOf(frontOfficeUrl) > -1) {
		towardUrl = towardUrl.substring(towardUrl.indexOf(frontOfficeUrl) + frontOfficeUrl.length, towardUrl.length);
	}
		//EURLEXNEW-3619: If the previous check doesn't match, there is also a case where the url has a prefix, but it is different from the frontOfficeUrl declared in the config*.properties files.
		//In this case, also check the url against the contextRoot and remove the prefix, if the url has one.
	//E.g. http://po-eurlexfo-test:7001/eurlex-frontoffice/relativeUrl
	else if (towardUrl.indexOf(contextRoot) > -1) {
		towardUrl = towardUrl.substring(towardUrl.indexOf(contextRoot) + contextRoot.length, towardUrl.length);
	}

	var callingUrl = pageUrl;
	if(queryString) {
		callingUrl += '?' + queryString;
	}

	url = authenticationRequiredUrl + '?towardUrl=' + encodeURIComponent(towardUrl) + '&callingUrl=' + encodeURIComponent(callingUrl);
	if(url.indexOf('isAjaxRequest=true') == -1) {
		url = url + (url.indexOf('?') == -1 ? '?' : '&') + 'isAjaxRequest=true';
	}

	loadModal(url, callback);
}


// Show the spinner and prepend an empty tempdata container div to the body so as to be used for loading the ajax response
function showHourglass(noHideOnClick){
// Show full page LoadingOverlay
	$.LoadingOverlay("show");
} // end showHourglass
// Remove the spinner and the tempdata container div which stores the result of an ajax request from the page.
function hideHourglass(){
	// Hide the full page spinner
	if($.LoadingOverlay){
		$.LoadingOverlay("hide");
	}
}

// The actual modal loading
function loadModal(url, callback) {
	// cleanup general modal in case there are remnants from previous modals
	$('#myModal .modal-body').empty().load(url + ' .alert, .SectionTitle, .ajaxContent', function (responseText, textStatus, jqXHR) {
		// In case of failure the user is informed and the spinner overlay is closed
		if (textStatus == "error") {
			alert("Request failed");
			hideHourglass();
		}
		else if (textStatus == "success") {
			fixModalContent();
			// Call the callback function - if one - and execute it.
			callFunction(callback);
			// TODO: Call focus as specified in getBootstrap model section
			// Shows up the loaded model
			// NOTE: see eurlex.js modal related code for catching the bs.show event and fixing modal size/location
			// Hide the spinner after the modal is properly loaded.
			hideHourglass();
			$('#myModal').modal({show: true});
		}
	});
}

// It performs any adhoc reconstruction to the modal content which is initially loaded as it comes
//from the ajax respose.
function fixModalContent() {
	// Move the title on modal-title and remove from modal content
	$('#myModal .modal-title').html($("#myModal .modal-body .SectionTitle").html());
	$("#myModal .modal-body .SectionTitle").remove();

	// In case the ajax content contains a smallModal class indicating that
	// the modal should be the small one and not the large then the class that makes it large is
	// removed from the modal-dialog.
	if ($("#myModal .smallModal").length > 0) {
		$("#myModal .modal-dialog").removeClass("modal-lg")
	} else {
		$("#myModal .modal-dialog").addClass("modal-lg")
	}

	// If there are alert meesages place them on top.
	if ($('#myModal .alert').length) {
		$('#myModal .alert').each(function() {
			$("#myModal .modal-body").prepend($(this));
		})
	}


	// Wrap the filterForm if it exists with a div containing the FixedModalContent class so to make its position fixed
	if (("#myModal .fixedModalFilterForm").length) {
		$("#myModal .fixedModalFilterForm").wrap("<div class='FixedModalContent'></div>");
		// Move the fixed modal content after the modal-title. Remove it from body
		$('#myModal .modal-header').after($("#myModal .modal-body .FixedModalContent"));
	}
	// Hide in modal
	$("#myModal .hideInModal").css({display:'none'});
	/* TODO Remove */
	/*  $(".onlyJsInline").css({display:'inline'});*/
}


// Hides the modal. After the hiding of the modal a clenup of the dom elements is triggered via
// the hidden.bs.modal event
function hideModal() {
	//EURLEXNEW-3817 - When IE 11 we load author modal as standalone page, like noJS case. However we will have JS. This is the case when the below is false
	if (!($(".modal-open").length == 0 && GetIEVersion() > 0)) { //Regular modals
		$("#CANCEL").attr("type", "button");	//to avoid form submission
		// Hide the modal
		$('#myModal').modal('hide');
	}
}
// Hides the modal by adding the dss display:none property. THis does not trigger a modal close event so no cleanup is happening, avoiding in that way cleaning the content of the next to display modal.
function hideButDontCloseModal() {
	hideHourglass();
	$('#myModal').hide();
}


// Restores to the default (empty elements) the general modal sections so as to clean the dom.
// The function is triggered once the modal is closed via the hidden.bs.modal bootstrap event.
function modalCleanup(){
	// Empty the reusable's modal sections before hide
	$("#myModal .modal-body").empty();
	$("#myModal .modal-title").empty();
	if ($('#myModal .modal-footer').length) {
		$('#myModal .modal-footer').empty()
	}
	// Remove the fixed modal content area
	if ($(".FixedModalContent").length) {
		$(".FixedModalContent").remove();
	}
}

/* MODAL - Ajax POST related functioanlity*/
// Perform Ajax post from a modal
function ajaxPostOnModal(url, formId, callback) {
	showHourglass();
	if(url.indexOf('isAjaxRequest=true') == -1) {
		url = url + (url.indexOf('?') == -1 ? '?' : '&') + 'isAjaxRequest=true';
	}
	$.ajax({
		type: "POST",
		url: url,
		data: $('#' + formId).serialize(),
		error: function () {
			alert("Request failed! ");
			hideHourglass();
		},
		success: function(data) {

			//EURLEXNEW-3973 : SET DISPLAY MSG FROM THE GIVEN OBJECT CREATED IN CONTROLLER
			//THE GIVEN OBJECT SHOULD HAVE ONLY A 'VALUE' PROPERTY WITH THE MSG,
			if($.isPlainObject(data) && data.hasOwnProperty('value')){
				data = data['value'];
			}

			// EURLEXNEW-3973: For EF surveys, reload the CAPTCHA after submission
			// so that a new one is available in case the user has provided wrong input.
			if (formId == 'experimentalFeaturesSurveyForm') {
				renderCaptcha();
			}

			// It is placed in a try/catch block so as to remove the spinner in case of a mistake and avoid blocking the page.
			try {
				// Create a dummy dom element
				var el = $( '<div></div>' );
				el.html(data);
				// Extract the main content from the dummy element
				var content;
				// Keep only the main Content in case it exists, alternatively,
				// when if it doesn't exist (in the case the view is rendered with loadViewWithoutTemplate) the entire response is retrieved
				if ($('.alert, .SectionTitle, .ajaxContent', el).length) {
					content = $('.alert, .SectionTitle, .ajaxContent', el);
				} else {
					// Whatever it returns
					content = el
				}

				$("#myModal .modal-body").html(content)
				fixModalContent();
				callFunction(callback);
				// Hide the spinner after the modal is properly loaded.
				hideHourglass();
				$('#myModal').modal({show: true});
			}
			catch(err) {
				alert("Exception: " + err.message)
				hideHourglass();
			}
		},
		complete: function (jqXHR, resultStr) {
			// In any case after close the spinner if the post has been complete.
			// NOTE: This is excecuted after the success and in case of failure too.
			hideHourglass();
		}
	});
}
/***** END OF MODALS *****/


// Performs an ajax POST request
function ajaxPost(url, formId) {
	$.ajax({
		type: "POST",
		url: url,
		data: $('#' + formId).serialize()
	});
}




/*Checboxes selection */
function selectAll(chkClass, updateSession, saveSessionPath, sessionContainerId){
	$("."+chkClass).each(function() {
		if(!this.checked) {
			$(this).prop('checked', true);
			if(updateSession == 'true') {
				updateDocumentToSave(this, saveSessionPath, sessionContainerId);
			}
		}
	});
}
function deselectAll(chkClass, updateSession, saveSessionPath, sessionContainerId){
	$("."+chkClass).each(function() {
		if(this.checked) {
			$(this).prop('checked', false);
			if(updateSession == 'true') {
				updateDocumentToSave(this, saveSessionPath, sessionContainerId);
			}
		}
	});
}
function toggleFullQuery(el){
	var fullQuery = $(el).next('.fullQuery');
	if($(fullQuery).css('display') == 'none') {
		$(el).html(hideQueryLabel);
		fullQuery.show();
	}
	else {
		$(el).html(viewQueryLabel);
		fullQuery.hide();
	}
}

function checkChildrenBoxes(el){
	if(el.checked) {
		$(el).parent("label").parent("li").find("input").each( function(){ if(!this.disabled) {
			$(this).prop('checked',true);
			synchroChk(this);
		}
		} );
	}
	else {
		$(el).parent("label").parent("li").find("input").each( function(){  if(!this.disabled) {
			$(this).removeAttr('checked');
			synchroChk(this);
		}} );
	}
}

function uncheckParentBox(el){
	if(!el.checked){
		var input = $(el).parent("label").parent("li").parent("div").parent("div").parent("ul").parent("li").children("label").children("input");
		$(input).prop('checked',false);
	}
}

/* AJAX Tree */
function loadNextLevel(el, path, queryString, callback){

	$.ajax({
		type: "GET",
		url: path,
		data: queryString + "&isAjaxRequest=true",
		success: function(msg){
			$(el).parent('li').append(msg);
			// EURLEXNEW-3970: ECB-Statistics widget should adjust
			// its height to that of ECB-Tree after new nodes are appended.
			if ($("#EcbMenuBlock1").length) {
				$("#EcbMenuBlock1").trigger("treeResize");
			}
			if ($("#IeniMenuBlock1").length) {
				$("#IeniMenuBlock1").trigger("treeResize");
			}
			callFunction(callback);
		}
	});

	// Fix the arrowState and make them collapsed or expanded in advance, before the ajax request is sent..
	if($(el).prop('nodeName').toLowerCase() == 'a') {
		if( $(el).children('img').attr('src').indexOf('expand') > 0)
			$(el).children('img').attr('src',imageMap['collapse-tree']);
		else
			$(el).children('img').attr('src',imageMap['expand-tree']);
	}
	else if($(el).prop('nodeName').toLowerCase() == 'input') {
		if( $(el).attr('src').indexOf('expand') > 0)
			$(el).attr('src',imageMap['collapse-tree']);
		else
			$(el).attr('src',imageMap['expand-tree']);
	}

	$(el).attr('onclick','');
	$(el).click(function() {
		toggleNextTreeLevel(this); return false;
	});
	return false;
}

function toggleNextTreeLevel(el) {

	/* TAS-1375 - <ul> is used for ECB tree and <ol> for IENI tree */
	if($(el).parent('li').children('ul,ol').length > 0){
		smartToogleTree($(el).parent('li').children('ul,ol'));
		// Fix the arrowState and make them collapsed or expanded.
		if($(el).prop('nodeName').toLowerCase() == 'a') {
			if( $(el).children('img').attr('src').indexOf('expand') > 0)
				$(el).children('img').attr('src',imageMap['collapse-tree']);
			else
				$(el).children('img').attr('src',imageMap['expand-tree']);
		}
		else if($(el).prop('nodeName').toLowerCase() == 'input') {
			if( $(el).attr('src').indexOf('expand') > 0)
				$(el).attr('src',imageMap['collapse-tree']);
			else
				$(el).attr('src',imageMap['expand-tree']);
		}
	}
}

function fillHierarchyTree() {
	var idValues = $("#idValues").val();
	var el = $("#" + idValues);
	var children = el.children();
	children.each(function(){
		var id = this.value;
		var escapedId= id.replace(/\./g,'')
		var chkElement= document.getElementById("chk_"+escapedId);
		if(chkElement){
			chkElement.checked = true;
		}
		selectHierarchyValue(chkElement);
	})
}

function switchDirectoryOfCaseLaw(id, value) {
	$("#" + id).attr('onclick', '').unbind('click');

	if(value == 'RJ') {
		$("#" + id).click(function() {
			showModal('next-tree-level.html?fromId=directoryCaseLawCode&fillType=fillHierarchyForm&code=RJ_1_CODED', 'fillHierarchyTree()', 'RJ,generatedHierarchyValues_DirCaseLaw'); return false;
		});
	}
	else{
		$("#" + id).click(function() {
			showModal('next-tree-level.html?fromId=directoryCaseLawCode&fillType=fillHierarchyForm&code=RJ_NEW_1_CODED', 'fillHierarchyTree()', 'RJ,generatedHierarchyValues_DirCaseLaw'); return false;
		});
	}
}

function filterAdvancedSearch() {
	var filter = sanitizeHtml($("#advancedFilter").val());
	var fromId = sanitizeHtml($("#fromId2").val());
	var qId = sanitizeHtml($("#qid2").val());
	var code = sanitizeHtml($("#code2").val());
	var idValues = sanitizeHtml($("#idValues2").val());
	var labelId = sanitizeHtml($("#labelId2").val());
	var oneElementPickField = sanitizeHtml($("#oneElementPickField2").val());
	var subdomain = sanitizeHtml($("#subdomain2").val());
	var hashTag = sanitizeHtml($("#hashTag2").val());

	var href = $("#filterForm").attr('action') + "&isAjaxRequest=true&filter=" + encodeURI(filter) + "&labelId=" + labelId + "&fromId=" + fromId + "&code="+code+"&idValues="+idValues+"&oneElementPickField="+oneElementPickField+"&subdomain="+subdomain+"&hashTag="+hashTag;
	if(typeof(qId)!='undefined') {
		href = href + "&qid=" + qId;
	}
	$("#referenceDataValues").empty();
	if(!$("#filter").children().last().is("img")){
		$("#referenceDataHelperAdvanced").append('<i id="loadingImg" class="fa fa-spinner fa-spin" aria-hidden="true"></i>');
		$("#referenceDataHelperAdvanced").addClass('text-center');
	}
	$("#referenceDataValues").load(href,
		function() {
			fillSelectedHierarchyValue();
			$("#referenceDataHelperAdvanced #loadingImg").remove();
			$("#referenceDataHelperAdvanced").removeClass('text-center');
		});
	$("#referenceDataValues").fadeIn();
}

function fillUrl(code) {
	var tab = code.split(",");
	var el = $("#" + tab[1]);
	var children = el.children();
	var idValues = "&idValues="+tab[1];
	if(tab.length >= 3) {
		idValues = idValues + "&labelId="+tab[2];
	}
	var data = "&values=";
	var flag = false;
	var metadataCode = tab[0];
	children.each(function(){
		if(this.value.indexOf(metadataCode) == 0) {
			flag = true;
			var tabValueIndex = this.value.indexOf("--");
			var checkedValue = this.value.substring(tabValueIndex + 2);
			data = data + checkedValue + ";";
		}
	})
	if(flag == true) {
		return idValues+data.substring(0, data.length-1);
	}
	else return idValues;
}
//select values
function selectHierarchyValue(checkbox) {
	if (checkbox) {
		if (checkbox.checked) {
			//avoid duplicate entries  for same id (i.e. Eurovoc)
			if ($('#selectedValues input[id='+checkbox.id+']').length == 0) {
				$("#selectedValues").append("<input type='checkbox' id='"+checkbox.id+"' value='"+checkbox.value+"' checked='checked'></input>");
				if ($("label[for='"+checkbox.id+"']").length==0){
					var label = $('div[for='+checkbox.id+']').first().text();
					$("#selectedValues").append("<label for='"+checkbox.id+"'>"+$.trim(label)+"</label>");
				}else
				{
					var label = $('label[for='+checkbox.id+']').first().text();
					var labelWithoutCode = label.replace(/\(.+?\)/g, "");
					$("#selectedValues").append("<label for='"+checkbox.id+"'>"+$.trim(labelWithoutCode)+"</label>");
				}
			}
			if ($(".modal-open").length == 0 && GetIEVersion() > 0) { //EURLEXNEW-3817 - When IE 11 we load author modal as standalone page, like noJS case. However here we do have JS.
				if ($("#deselectedValues #" + checkbox.id).length > 0) {
					$("#deselectedValues #" + checkbox.id).remove();
				}
			}
		} else {
			if ($("#referenceDataValues input[id='"+ checkbox.id +"']:checked").length == 0) {
				$("#selectedValues input[id='"+ checkbox.id +"']").remove();
				$("#selectedValues label[for='"+ checkbox.id +"']").remove();
			}
			if ($(".modal-open").length == 0 && GetIEVersion() > 0) { //EURLEXNEW-3817 - When IE 11 we load author modal as standalone page, like noJS case. However here we do have JS.
				if ($("#deselectedValues #" + checkbox.id).length == 0) {
					$("#deselectedValues").append("<input type='checkbox' id='" + checkbox.id + "' value='" + checkbox.value + "' checked='unchecked'></input>");
				}
			}
		}
	}
}

function fillSelectedHierarchyValue() {
	$("#selectedValues input[type='checkbox']").each(function(){
		if(this.checked) {
			var chk = $("input[id='"+this.id+"']");
			$(chk).prop('checked',true);
		}
	});
}

function fillSimplePickField(value) {
	var fromID = $("#fromId").val();
	$("#"+fromID).val(value);
	hideModal();
	return false;
}

function fillEurovocForm_js(){
	var val="";
	var fromID = $("#fromId").val();
	var idValues = $("#idValues").val();
	var prefix = ($("#prefix").val());

	if ($("#"+fromID).prop('nodeName').toLowerCase() == 'input'){
		var array = [];
		$("#selectedValues input[type='checkbox']").each(function(){
			if(this.checked) {
				var labelVal = $('#selectedValues label[for='+this.id+']').text();
				var indexNt = labelVal.indexOf('NT');
				if(indexNt != -1) {
					var indexSpace = labelVal.indexOf(' ');
					labelVal = labelVal.substring(indexSpace + 1);
				}
				if($.inArray(labelVal, array) == -1) {
					array.push(labelVal);
				}
			}
		})

		// surround label with double quotes
		for (var i=0; i<array.length; i++) {
			val += "\"" + array[i] + "\"; ";
		}
		val = val.substring(0, val.length -2);
		$("#"+fromID).val(val);

		var options = $("#"+idValues + " option[value*='"+prefix+"']");
		options.remove();
		$("#selectedValues input[type='checkbox']").each(function(){
			if(this.checked) {
				var tabValueIndex = this.value.indexOf("--");
				var checkedValue = this.value.substring(tabValueIndex + 2);
				if ($('#'+idValues+' option[value='+this.value+']').length == 0) { // avoid
					// duplicate
					// entry
					// with
					// same
					// id
					// (i.e.
					// Eurovoc)
					$("#"+idValues).append("<option value='"+ this.value +"' selected='selected'></option>");
				}
			}
		})
	}
	hideModal();
}

function fillHierarchyFormInsCited_js(){
	var checked = false;
	$(".ajaxContent input[type='checkbox']").each(function(){
		if(this.checked) {
			checked = true;
			return false;
		}
	})
	fillHierarchyForm_js();
	if(checked) {
		$("#instrumentCitedTreaties").prop('checked',true);
	}
}

function fillSummaryCodeForm_js(){
	fillHierarchyForm_js();
}

function fillHierarchyForm_js(){
	var val="";
	var fromID = $("#fromId").val();
	var idValues = $("#idValues").val();
	var prefix = ($("#prefix").val());

	if ($(".modal-open").length == 0 && GetIEVersion() > 0) { //EURLEXNEW-3817 - When IE 11 we load author modal as standalone page, like noJS case. However here we do have JS.
		var selectionsFromServer = $("#selectedFromServer").html().trim().replace("[","").replace("]","").split(", ") //selectedFromServer contains all possible selections made before modal was opened.
		$.each(selectionsFromServer, function( index, value ) {
			if (value !="" && $("#deselectedValues input#chk_AU_CODED--" + value + "").length  == 0) { //unless it was deselected before, add it as current selection
				$("#referenceDataValues .col-sm-12").append("<input name='values' type='checkbox' class='hidden' id='chk_AU_CODED--" + value + "' value='AU_CODED--" + value + "'>");
				$("#chk_AU_CODED--" + value).prop( "checked", true );
			}
		});
		$("#selectedValues input[type='checkbox']").each(function() { //selectedValues contains all selections made before and after filter
			var InputId = $(this).attr("id");
			var InputValue = $(this).attr("value");
			if (InputValue != "" && this.checked && $("input[value='" + InputValue + "'][name='values']").length == 0) {	//If input with name=values is missing because of filtering, then add it
				$("#referenceDataValues .col-sm-12").append("<input name='values' type='checkbox' class='hidden' id='" + InputId + "' value='" + InputValue + "'>");
				$("#" + InputId).prop( "checked", true );
			}
		});
	} else {	//When regular modal
		$("input[name*=SUBMIT_]").attr("type", "button");	//to avoid form submission
		//OLD LOGIC - before EURLEXNEW-3817
		if($("#"+fromID).prop('nodeName').toLowerCase() == 'textarea') {
			$("#selectedValues input[type='checkbox']").each(
				function(){ if(this.checked) val += $('#selectedValues label[for='+this.id+']').text() +"\r\n"; })
			val = val.substring(0, val.length -2);
			$("#"+fromID).val(val);
		}
		else if ($("#"+fromID).prop('nodeName').toLowerCase() == 'input'){
			var options = $("#"+idValues + " option[value*='"+prefix+"']");
			options.remove();
			$("#selectedValues input[type='checkbox']").each(function(){
				if(this.checked) {
					var tabValueIndex = this.value.indexOf("--");
					var checkedValue = this.value.substring(tabValueIndex + 2);
					val += $('#selectedValues label[for='+this.id+']').text()+"; ";
					// val += getHierarchySelectedValue(checkedValue)+"; ";
					$("#"+idValues).append("<option value='"+ this.value +"' selected='selected'></option>")
				}
			})
			val = val.substring(0, val.length -2);
			$("#"+fromID).val(val);
		}
		else {
			$("#selectedValues input[type='checkbox']").each(function(){
				if (this.checked) {
					val += this.value+"; ";
				}
			})
			val = val.substring(0, val.length -2);
			$("#"+fromID).text(val);
		}
		$("#"+fromID).change();
		hideModal();
	}
}

function fillHierarchyFormAuthorEli_js(){
	fillHierarchyForm_js();
}

function clearHierarchyForm_js() {
	$("#selectedValues").empty();
	$(".ajaxContent input[type='checkbox']").each(
		function(){
			if ($(".modal-open").length == 0 && GetIEVersion() > 0) { //EURLEXNEW-3817 - When IE 11 we load author modal as standalone page, like noJS case. However here we do have JS.
				if (this.checked == true && $("#deselectedValues #" + $(this).attr("id")).length == 0) {	//if previously checked, put in deselected values
					$("#deselectedValues").append("<input type='checkbox' id='" + $(this).attr("id") + "' value='" + $(this).attr("value") + "' checked='unchecked'></input>");
				}
			}
			this.checked = false;
		}
	);
}

function fillForm(){
	var val="";
	var fromID = $("#fromID").val();
	$(".ajaxContent input[type='checkbox']").each(function(){ if(this.checked) val += this.value+"; "; })
	val = val.substring(0, val.length -2);
	if ($("#"+fromID).prop('nodeName').toLowerCase() != 'input'){
		$("#"+fromID).text(val);
	}
	else{
		$("#"+fromID).val(val);
	}

	hideModal();
}

var baseHref;
function updateHref(el, chkClass, link){
	$(".alert-success").addClass("hidden");
	$(".alert-warning").addClass("hidden");

	if(!baseHref) {
		baseHref = $(el).attr('href');
	}

	var suite = "";
	$("."+chkClass).each(
		function(){
			if(this.checked) {
				var id = this.value.replace(':', '_');
				var fullUrl = $("#"+id).attr('name');
				if (fullUrl == '') {
					var str = link.replace(/{v}/g, this.value);
					suite= suite + encodeURIComponent(str) + "%0A";
				}
				else{
					suite= suite + encodeURIComponent(fullUrl) + "%0A";
				}
			}
		} );

	if (suite.length) {
		var href = baseHref + "%0A" + suite;
		$(el).attr('href', href);
	}
	else{
		$(".alert-warning").removeClass("hidden");
		$(".alert-success").addClass("hidden");
		$("#warningExp").addClass("hidden");
		$("#warningNoCheckboxSelected").addClass("hidden");

		$(el).removeAttr('href');
		$(el).attr('href','javascript:void(0);');
	}
}

function loadMetadataCheckBoxes(){
	$("input[id^='advanced']").each( function(){
		synchroChk(this);

	});
}
function showTab(el,linkedId, parent){
	var parentEl;
	if(parent!=null)
		parentEl="#"+parent;
	else
		parentEl='body';


	if($(parentEl+" #"+linkedId).css('display')=='block'){
		$("#"+linkedId).focus();
		return;
	}
	$(parentEl+" .active").removeClass("active");
	$(el).addClass("active");
	$(parentEl+" .tab-pane").hide();
	$(parentEl+" #"+linkedId).show();
	$("#"+linkedId).focus();
}
function collapseAdvancedEmptyMetadataGroup(){
	// ELX-1364
	$("#tabs-5 .simpleJsTree a.expandLink").each(function() {
		var ulElement = $(this).parent('li').find('ul:first');
		// verify if there are checked boxes for that category
		var nbChecked = ulElement.parent('li').find('input:checked').length;
		if (nbChecked == 0 && ulElement.is(':visible')) {
			// collapse empty categories
			$(this).click();
		}
	});
}

function synchroChk(el){
	var typ = 'advanced';
	if( (el.id).indexOf('advanced') > -1 )
		typ='simple';
	if (el.checked){
		$("#"+typ+(el.id).substring( (el.id).indexOf('_'), (el.id).length )).prop('checked',true);
	}
	else{
		$("#"+typ+(el.id).substring( (el.id).indexOf('_'), (el.id).length )).prop('checked',false);
		uncheckParentBox(el);
		uncheckParentBox($("#"+typ+(el.id).substring( (el.id).indexOf('_'), (el.id).length )));
	}
}

function loadMetadataCheckBoxesSimple(){
	$("input[id^='simple']").each( function(){
		synchroChk(this);

	});
}

function switchDomainSelection(){
	$("input[id^='chkMultiple']").each( function(){
		if (this.checked){
			$("#"+this.id).removeAttr('checked');
		}

	});
	$("#multipleDomain, #normalDomain").toggle();
}
function switchTextSearch(option){
	if(option == "more") {
		$("label[for='fullText']").html(allTheseWordsLabel);
	}
	else {
		$("label[for='fullText']").html(fullTextLabel);
	}
	$("#moreTextSearch, #normalTextSearch").toggle();
}

function addFavorite(url,name,errorUrl){
	var callback = $(this).data('callback');
	url = removeAttributes(url,'qid,rid');
	if(name==null || url==null){
		showModal(encodeURI(errorUrl),callback);
		return;
	}
	if (url != null && url.indexOf("http:") < 0){
		showModal(encodeURI(errorUrl),callback);
		return;
	}
	if(window.external && !window.sidebar && navigator.userAgent.toLowerCase().indexOf('chrome')< 0) { // add
		// IE
		// favorite
		external.AddFavorite(url,name);
	} else if(window.sidebar && sidebar.addPanel && navigator.userAgent.toLowerCase().indexOf('chrome')< 0) { // add
		// to
		// FF
		// bookmarks
		sidebar.addPanel(name,url,'');
	} else {   // unknown browser: report user
		showModal(encodeURI(errorUrl),callback);
	}
}

function updateDocumentToSave(ob, qid, url) {
	var arg;
	if(ob.checked) {
		arg = "action=addDocToSave";
	}else {
		arg = "action=removeDocToSave"
	}
	arg = arg + "&legalContentId=" + ob.value + "&" + qid + "&isAjaxRequest=true";
	$.ajax({
		type: "GET",
		url: url,
		data:arg,
		success: function(msg){
		}
	});

	processClearButtonAccordingSelectedItems();
}

function processClearButtonAccordingSelectedItems(){

    if(Array.from(document.querySelectorAll('[id^="selectedDocument"]')).filter(item => item.checked).length > 0){

        var btn = document.getElementById("clearSelectedCheckboxes_1");
        btn.parentElement.className = btn.parentElement.className.replaceAll(" hidden","");

        var btnBottom = document.getElementById("clearSelectedCheckboxes_2");
        btnBottom.parentElement.className = btnBottom.parentElement.className.replaceAll(" hidden","");

    }else{
        var btn = document.getElementById("clearSelectedCheckboxes_1");
        btn.parentElement.className = btn.parentElement.className + " hidden";

        var btnBottom = document.getElementById("clearSelectedCheckboxes_2");
        btnBottom.parentElement.className = btnBottom.parentElement.className + " hidden";

    }

}

Date.prototype.toSimpleString=function(){
	return this.getDate()+'/'+(this.getMonth()+1 < 10 ? '0':'')+(this.getMonth()+1)+'/'+this.getFullYear();
}

function collapseAllTable() {
	$("a[id^='toggleTHead']").each(function() {
		if($(this).children('img').first().attr('src').indexOf("minimize")>-1) {
			$(this).click();
		}
	});
}

function expandAllTable() {
	$("a[id^='toggleTHead']").each(function() {
		if($(this).children('img').first().attr('src').indexOf("maximize")>-1) {
			$(this).click();
		}
	});
}
function selectRadioButton(id,el) {
	if(($(el).prop('nodeName').toLowerCase() == 'input' && el.value != '' & typeof(el.value) != 'undefined') || ($(el).prop('nodeName').toLowerCase() == 'select' && el.value != 'ALL')){
		$("#"+id).prop('checked',true);
	}
}
var radioStatus;
function selectRadioButtonGlobal(id,el) {
	if(($(el).prop('nodeName').toLowerCase() == 'input' && el.value != '' & typeof(el.value) != 'undefined') || ($(el).prop('nodeName').toLowerCase() == 'select' && el.value != 'ALL')){
		if(el.value != radioStatus) {
			$("#"+id).prop('checked',true);
		}
	}
}

function checkDocuments(el, saveSessionPath, sessionContainerId){
	$('input[id^="selectedDocument"]').each(function() {
		if(el.checked) {
			if(!this.checked) {
				$(this).prop("checked", true);
				$(".ResultsTools input[type='checkbox'].hidden-sm").prop("checked", true);	//to also check the other 'select all' checkbox
				updateDocumentToSave(this, saveSessionPath, sessionContainerId);
			}
		} else {
			if(this.checked) {
				$(this).prop("checked", false);
				$(".ResultsTools input[type='checkbox'].hidden-sm").prop("checked", false); //to also un-check the other 'select all' checkbox
				updateDocumentToSave(this, saveSessionPath, sessionContainerId);
			}
		}
	});
}

function toggleHistoricalElements(el,id) {
	$("#"+id).children("span[id^='expElement_']").each(
		function() {
			var classEl = $(this).attr('class');
			if(classEl.indexOf('extended')>-1) {
				$(this).attr('style','display:none');
				$(this).attr('class', 'hideInJsInline');
			} else {
				$(this).attr('style','display:inline');
				$(this).attr('class', classEl + ' extended');
			}
		}
	);
	if($(el).children('img').attr('src') != null) {
		if( $(el).children('img').attr('src').indexOf('maximize') > -1) {
			$(el).children('img').attr('src',imageMap['box-minimize']);
			$(el).children('img').attr('alt', '-');
		}
		else {
			$(el).children('img').attr('src',imageMap['box-maximize']);
			$(el).children('img').attr('alt', '+');
		}
	}
}

function collapseAllHistoricalElements() {
	$("a[id$='HistProcHref']").each(function() {
		if($(this).children('img').first().attr('src').indexOf("minimize")>-1) {
			$(this).click();
		}
	});
}

function expandAllHistoricalElements() {
	$("a[id$='HistProcHref']").each(function() {
		if($(this).children('img').first().attr('src').indexOf("maximize")>-1) {
			$(this).click();
		}
	});
}

function submitPaging(el, nbreOfPage) {
	try {
		if($(el).val() == '') {
			$(el).attr('class', 'paging');
			return;
		}
		var currentPage = parseInt($(el).val());
		if(parseFloat($(el).val()) != currentPage || isNaN(currentPage) || endWith($(el).val(), '.')) {
			$(el).attr('class', 'paging pagingError');
			return false;
		}
		var maxPage = parseInt(nbreOfPage);
		if(currentPage > maxPage || currentPage == 0) {
			$(el).attr('class', 'paging pagingError');
			return false
		} else {
			$(el).attr('class', 'paging');
			return true;
		}
	} catch (e) {
		$(el).attr('class', 'paging pagingError');
		return false;
	}
}

function checkNumeric(e) {
	var k = e.keyCode;
	if( e.which ){
		k = e.which;
	}
	if(k != '48' && k != '49' && k != '50' && k != '51' && k != '52' && k != '53' && k != '54' && k != '55' && k != '56' && k != '57'
		&& k != '13' && k != '8' && k != '37' && k != '39' && k != '9' && k != '46'){
		if(e.preventDefault) {
			e.preventDefault();
		}
		else {
			event.returnValue = false;
		}
	}
}

function checkPaging(el, nbreOfPage) {
	try {
		if($(el).val() == '') {
			$(el).attr('class', 'paging');
			return;
		}
		var currentPage = parseInt($(el).val());
		if(parseFloat($(el).val()) != currentPage || isNaN(currentPage) || endWith($(el).val(), '.')) {
			$(el).attr('class', 'paging pagingError');
			return;
		}
		var maxPage = parseInt(nbreOfPage);
		if(currentPage > maxPage || currentPage == 0) {
			$(el).attr('class', 'paging pagingError');
		} else {
			$(el).attr('class', 'paging');
		}
	} catch (e) {
		$(el).attr('class', 'paging pagingError');
	}
}

function endWith(val, endStr) {
	var lastIndex = val.lastIndexOf(endStr);
	return (lastIndex != -1) && (lastIndex + endStr.length == val.length);
}

// INPUT FILTER
function fadeOut(id) {
	if($("#"+id).is(":visible")) {
		$("#"+id).fadeTo('fast', 0, function() {
			$(this).css({visibility: "hidden"});
		});
	}
}

function fadeIn(id) {
	if($("#"+id).is(":hidden") || $("#"+id).css("visibility") == 'hidden') {
		$("#"+id).fadeTo('fast', 1, function() {
			$(this).css({visibility: "visible"});
		});
	}
}

function advancedSearchKeyUp(input, imageId) {
	if ($.trim($("#"+input).val()) == "") {
		hideClearSearchFieldButton(imageId);
	} else {
		showClearSearchFieldButton(input, imageId);
	}
}

function clearAdvancedSearchClick(input, imageId, clearButtonId) {
	hideClearSearchFieldButton(imageId);
	$("#"+input).val('');
	$("#"+clearButtonId).click();
}

function hideClearSearchFieldButton(imageId) {
	fadeOut(imageId);
}

function showClearSearchFieldButton(input, imageId) {
	if ($.trim($("#"+input).val()) != "") {
		fadeIn(imageId);
		/*Set the right margin of the clear button dynamically so that it doesn't overlap 
		 * with text, as the following button's width varies with the interface language.*/
		var filterBtnWidth = $("#"+imageId).next().width();
		var right = filterBtnWidth + $("#"+imageId).width();
		$("#"+imageId).css("right", right);
	}
}
// END INPUT FILTER

function selectAllPreferences(id, select) {
	$('input[id^="'+id+'"]').each(function() {
		if(select == 'true') {
			$(this).prop('checked', true);
		} else {
			$(this).prop('checked', false);
		}
	});
}

//notice
function instrumentInvolvedChanged(checkBox, input) {
	if (checkBox.checked) {
		$("#"+input).attr('disabled', false);
	} else {
		$("#"+input).attr('disabled', true);
	}
}


// Reset a list of fields to default value.
// @param fields the list of fields to reset
function resetFields(fields) {
	for (var i=0;i<fields.length;i++) {
		var field = $("#" + fields[i]);
		if (field.is("select")) { // drop down
			field.prop('selectedIndex',0);
		}
		else if (field.is("input")) { // input field
			field.prop("value",'');
		}
	}
}

function resetInputFieldsUnchecked() {
	var inputs = $('#document-reference-fields input:not([type=hidden]');
	for(var i =0; i < inputs.length; i++) {
		if(inputs[i].id != 'typeOfActStatusAll') {
			$(inputs[i]).prop("checked",false);
		}
	}
}

// Enable a list of fields.
// @param fields the list of fields to enable
function enableFields(fields) {
	for (i=0;i<fields.length;i++) {
		var field = $("#" + fields[i]);
		field.attr('disabled', false);
	}
}

// Disable a list of fields.
// @param fields the list of fields to disable
function disableFields(fields) {
	for (i=0;i<fields.length;i++) {
		var field = $("#" + fields[i]);
		field.attr('disabled', true);
	}
}

function initDatePicker(locale) {
	//Temporary solution for Gaeilge - Irish language (not found in bootstrap calendar)
	var locale = locale =='ga' ? "en-GB" : locale;
	// Check if numeric
	var regExp = /[0-9\/]/;
	var ctrlKey = 17;
	//Valid input types for calendar
	$("input[id^='dateExact'], input[id^='dateFrom'], input[id^='dateTo'], #exDateField, #rangeFromDateField, #rangeToDateField").on('keydown keyup', function(e) {
		var value = String.fromCharCode(e.which) || e.key;
		if (!regExp.test(value)
			&& e.which != 191 // forward slash
			&& e.which != 111 // forward slash (numpad-divide)
			&& e.which != 8   // backspace
			&& e.which != 46  // delete
			&& e.which != 13  // enter
			&& e.which != 9  // tab
			&& e.which != 37 // arrow left
			&& e.which != 38 // arrow up
			&& e.which != 39 // arrow right
			&& e.which != 40 // arrow down
			&& ((e.which != 65 || e.which != 86 || e.which != 67) && (e.ctrlKey === false)) // Ctrl+A, Ctrl+C, Ctrl+V
			&& (e.which < 96  // numpad keys
				|| e.which > 105)
		) {
			e.preventDefault();
			return false;
		}
		if(e.shiftKey) { e.preventDefault(); }
	});

	//Initialization of datepicker
	$("[id^=CalendardateExact], [id^=CalendardateFrom],[id^=CalendardateTo]").datetimepicker({
		format: 'DD/MM/YYYY', 							//Set calendar format
		useStrict: true,								//Strict date parsing when considering a date to be valid
		useCurrent: false,								//Not set the picker to the current date
		keepInvalid: true,								//Acceptable formats
		showTodayButton: true, 							//Today icon
		showClose: true,								//Close icon
		keepOpen: false,								//Will cause the date picker to stay open after selecting a date if no time components are being used
		locale:locale,								 	//Current language
		allowInputToggle: true,							//Toggles the calendar widget
		//Translations for datepicker so as to override default plugin tooltips
		tooltips: {today:getWTLabel('today'), close:getWTLabel('close'), selectMonth:getWTLabel('selectMonth'), prevMonth:getWTLabel('prevMonth'), nextMonth:getWTLabel('nextMonth'),
			selectYear:getWTLabel('selectYear'), prevYear:getWTLabel('prevYear'), nextYear:getWTLabel('nextYear'),
			prevDecade:getWTLabel('prevDecade'), nextDecade:getWTLabel('nextDecade')},


	}).on('dp.show dp.update', function () {
		//Disable century view mode from calendar
		$(".datepicker-years .picker-switch").removeAttr('title')
			.on('click', function (e) {
				e.stopPropagation();
			});
	});
	//Range Date - minimum date 	
	$("[id^=CalendardateFrom]").on("dp.show",function(e) {
		var calendarId = e.target.id;
		var calendarSuffix = calendarId.slice(16);
		var selectedDate = $("#CalendardateTo" + calendarSuffix).data("DateTimePicker").date();
		var calendarInputId = $("#dateTo"+calendarSuffix+", #rangeToDateField" ).attr('id');
		var inputVal = document.getElementById(calendarInputId).value;
		if ((!selectedDate) || (inputVal == "")) {
			$("#CalendardateFrom"+calendarSuffix).data("DateTimePicker").maxDate(false);
		} else {
			$("#CalendardateFrom"+calendarSuffix).data("DateTimePicker").maxDate(selectedDate);
		}
	});
	// Range Date - maximum date
	$("[id^=CalendardateTo]").on("dp.show",function(e) {
		var calendarId = e.target.id;
		var calendarSuffix = calendarId.slice(14);
		var selectedDate = $("#CalendardateFrom"+calendarSuffix).data("DateTimePicker").date();
		var calendarInputId = $("#dateFrom"+calendarSuffix+", #rangeFromDateField").attr('id');
		var inputVal = document.getElementById(calendarInputId).value;
		if ((!selectedDate) || (inputVal == "")) {
			$("#CalendardateTo"+calendarSuffix).data("DateTimePicker").minDate(false);
		} else {
			$("#CalendardateTo"+calendarSuffix).data("DateTimePicker").minDate(selectedDate);
		}
	});
	//Clear input content when type on another input
	$("input[id^='dateExact']").on('blur keyup',function(e) {
		tmpval = $(this).val();
		if (tmpval != '') {
			var idValue = $(this).prop("id");
			var IdRadio = idValue.replace("Exact", "Specific");
			$("input[id=" + IdRadio + "][value='SPECIFIC']").prop("checked", true);
			resetFields([ idValue.replace("Exact", "From"),
				idValue.replace("Exact", "To") ]);
		}
	});

	$("input[id^='dateFrom']").on('blur keyup', function(e) {
		tmpval = $(this).val();
		if(tmpval != '') {
			var idValue = $(this).prop("id");
			var IdRadio=idValue.replace("From", "Range");
			$("input[id="+IdRadio+"][value='RANGE']").prop("checked",true);
			resetFields([idValue.replace("From", "Exact")]);
		}
	});

	$("input[id^='dateTo']").on('blur keyup', function(e) {
		tmpval = $(this).val();
		if(tmpval != '') {
			var idValue = $(this).prop("id");
			var IdRadio=idValue.replace("To", "Range");
			$("input[id="+IdRadio+"][value='RANGE']").prop("checked",true);
			resetFields([idValue.replace("To", "Exact")]);
		}
	});

	//Clear input content when click on another radio button
	$("[id^=dateSpecific]").click(function() {
		var idValue = $(this).attr("id");
		resetFields([idValue.replace("Specific", "From"), idValue.replace("Specific", "To")]);
	});

	$("[id^=dateRange]").click(function() {
		var idValue = $(this).attr("id");
		resetFields([idValue.replace("Range", "Exact")]);
	});


}

// TODO to be removed?
function showPlaceholderIfEmpty(input) {
	if (navigator.userAgent.indexOf('MSIE 7')!=-1 || navigator.userAgent.indexOf('MSIE 8')!=-1 || navigator.userAgent.indexOf('MSIE 9')!=-1){
		if( input.val() === '' ){
			input.data('placeholder').removeClass('placeholder-hide-except-screenreader');
		}else{
			input.data('placeholder').addClass('placeholder-hide-except-screenreader');
		}
	}
}

function removeAttributes(url, csvAttributeNames) {
	var attributeNames = csvAttributeNames.split(',');
	var array = url.split("?");
	var pageUrl = array[0];
	var queryString = array[1];

	if (queryString && queryString != "") {
		array = queryString.split("&");
		for (var i = array.length - 1; i >= 0; i -= 1) {
			param = array[i].split("=")[0];
			if ($.inArray(param, attributeNames) != -1) {
				array.splice(i, 1);
			}
		}

		if(array.length >= 1) {
			pageUrl += "?" + array.join("&");
		}
	}

	return pageUrl;
}

function checkUploadFileSize(inputId, errorId) {
	var input = document.getElementById(inputId+"_js");
	if(input.files != undefined && input.files[0] != undefined) {
		if(input.files[0].size > maxUploadSize) {
			$("#"+errorId).addClass('has-error');
			$("#input-file").append('<span id=\"'+inputId+'.errors\" class=\"help-block\">' + fileTooBigLabel + '</span>');
			return false;
		}
	}
	return true;
}


function showAllMneTransposedDir(caller){
	$(caller).addClass('hidden');
	var curElem = $(caller);
	while(curElem.next().hasClass('hiddenTransposedDir')){
		curElem = curElem.next();
		curElem.removeClass('hidden');
	}
	curElem.next().removeClass('hidden');
}

function showLessMneTransposedDir(caller){
	$(caller).addClass('hidden');
	var curElem = $(caller);
	while(curElem.prev().hasClass('hiddenTransposedDir')){
		curElem = curElem.prev();
		curElem.addClass('hidden');
	}
	curElem.prev().removeClass('hidden');
}

function zoom(element, selectDocumentColumnId, cellarId, qid) {
	var spinner = '<img id=\"loadingImg\" class=\"printNone\" src=\"' + imageMap['ajax-loader'] + '\" title=\"' + zoomingTextMetadataLabel + '\" alt=\"' + zoomingTextMetadataLabel + '\" />';
	var rowspan = $('#'+selectDocumentColumnId).prop('rowspan');
	$('#'+selectDocumentColumnId).attr('rowspan', rowspan + 1);
	$('tr.zoomHidden td').first().append(spinner);
	$('tr.zoomHidden').first().removeClass('zoomHidden');
	$.ajax({type: 'GET',
		url: 'zoom.html',
		data: { cellarId: cellarId, qid: qid },
		timeout: 90000,
		cache: false,
		success: function(json) {
			if(json) {
				$('tr.zoom td').first().empty();
				$.each(json.zoom, function(izoom, zoom) {
					var onematch = false;
					$.each(zoom.content, function(icontent, content) {
						var value = content.value.replace(/&lt;em&gt;/g,"<em>").replace(/&lt;\/em&gt;/g,"</em>");
						if(!onematch && value) {
							$('tr.zoom td').first().append('<ul><li class=\"zoom-metadata\">' + zoom.metadata + '</li></ul>');
							onematch = true;
						}

						if(value){
							$('tr.zoom td ul').last().append('<li lang="' + json.language + '">' + '... ' + value + ' ...' + '</li>');
						}
					});
				});
			}
			$(element).remove();
			if ($('tr.zoom td').first().text()) {
				$('tr.zoom td').first().addClass('zoomed');
			}  else {
				$('#'+selectDocumentColumnId).attr('rowspan', rowspan);
				$('tr.zoom').first().addClass('hidden');
			}
			$('tr.zoom').first().removeClass('zoom');
			$('a.zoomHidden').first().click(); // next zoom
		},
		error: function() {
			$(element).remove();
			$('tr.zoom td').first().empty();
			$('tr.zoom td').first().append(
			        '<ul><li class=\"zoom-metadata\"><span class=\"alert alert-danger userMsgSimple\" role=\"alert\"><span class=\"fa fa-exclamation-circle\" aria-hidden=\"true\">&nbsp;</span>'
			        + zoomingErrorLabel + '</span></li></ul>');
			$('tr.zoom td').first().addClass('zoomed');
			$('tr.zoom').first().removeClass('zoom');
			$('a.zoomHidden').first().click(); // next zoom
		}
	});
}

function escapeHtml(string) {
	var entityMap = {"&": "&amp;","<": "&lt;",">": "&gt;",'"': '&quot;',"'": '&#39;',"/": '&#x2F;'};
	return String(string).replace(/[&<>"'\/]/g, function (s) {
		return entityMap[s];
	});
}

function addAnchorOnSubmit(element, id){
	var form = $(element).closest('form')
	form.prop('action',form.prop('action')+'#'+id);
}

//returns IE versions or "OTHER" for rest browsers
//currently not supported browsers IE older than 11
function hasBrowserVersion(){
	//IE < VERSION = 11
	var msie = window.navigator.userAgent.indexOf('MSIE ');
	//IE == VERSION = 7
	var msie7 = window.navigator.userAgent.indexOf('MSIE 7');
	//IE == VERSION = 11
	//var msie11 = !!navigator.userAgent.match(/Trident.*rv\:11\./);
	if (msie7>0) return "IE7";
	else if (msie>0) return "IE";
	//else if (msie11>0) return "IE11";
	else return "OTHER";
}

//creates user message in case of not supported browser
function createUserMsgIncompatibleBrowser(){
	// if no cookie was created to store preference and if browser version 7 < IE < 11
	if (getCookie('incompatibleBrowser') == null && hasBrowserVersion() == 'IE'){
		var browserNotSupportedClass = $('#browserNotSupported');
		var cookieMsgClass = $('#usermsgCookie');
		if (browserNotSupportedClass.length) {
			browserNotSupportedClass.css("display", "inherit");//display message
			if (cookieMsgClass.length>0){//fix css
				cookieMsgClass.css("margin-bottom","0px");
				browserNotSupportedClass.css("margin-bottom","20px");
			}
		}
	}
	//only javascript methods supported by IE7
	else if (getCookie('incompatibleBrowser') == null && hasBrowserVersion() == 'IE7'){
		var browserNotSupportedId = document.getElementById('browserNotSupported');
		var cookieMsgId = document.getElementById('usermsgCookie');
		if (browserNotSupportedId.all.length>0) {
			browserNotSupportedId.style.cssText = "DISPLAY: inherit";//display message
			if (cookieMsgId.all.length>0){//fix css
				cookieMsgId.style.marginBottom = "0px";
				browserNotSupportedId.style.marginBottom = "20px";
			}
		}
	}
}

//hides message and creates cookie to store preference if user clicks Dismiss link
function dismissUserMsg(){
	if (hasBrowserVersion() == 'IE7'){// IE7 compatible js code
		document.getElementById('browserNotSupported').style.cssText = "DISPLAY: none";
		document.getElementById('usermsgCookie').style.marginBottom = "20px";
	}else{//other IE versions
		$('#browserNotSupported').hide(500);
		$('#usermsgCookie').css("margin-bottom","20px");
	}
	createCookie("incompatibleBrowser","true",30);
}

function filterList(ulSelector, filterString, applyFilterAtDepth) {
	if(!applyFilterAtDepth) {
		applyFilterAtDepth = 0;
	}

	var newFilterString = escapeRegexExceptStarAndQuestionMark(filterString).replace("?", ".?").replace("*", ".*");
	var exp = new RegExp(newFilterString,"gi");
	ulSelector.find('li').show();

	var ulLevelSelector = ulSelector;
	for(var i = 0; i < applyFilterAtDepth; i++) {
		ulLevelSelector = ulLevelSelector.find('li').find('ul');
	}

	var lis = ulLevelSelector.find('li.leaf');
	var allLis = ulSelector.find('li');

	lis.removeClass('matched');
	allLis.hide();

	lis.children('a, span, label').filter(function(index, element) {
		return $(element).text().match(exp);
	}).parent().addClass('matched');

	allLis.filter('.matched, :has(li.matched)').show();

	highlight(ulLevelSelector.find('li.leaf.matched'), newFilterString);
}
function highlight(selector, regexpString) {
	var leafText = selector.find('label span.translatedText, a, span');

	leafText.each(function(index, element) {
		var html = $(element)[0].innerHTML;
		var newHtml = replaceAllIgnoreCase(replaceAllIgnoreCase(html, "<em>", ""), "</em>", "");
		$(element).html(newHtml);
	});

	if(!isBlank(regexpString)) {
		var exp = new RegExp("("+regexpString+")(?![^<]*>)","gi");
		leafText.each(function(index, element) {
			var html = $(element)[0].innerHTML;
			var newHtml = html.replace(exp, "<em>$&</em>");
			var newElem = $(element).html(newHtml);
		});
	}
}
//remove em tag (highlight) - use this when user clicks close icon to reset filter searches
function removeHighlight(el) {
	$(el+' em').contents().unwrap();
}
function escapeRegexExceptStarAndQuestionMark(str) {
	return (str+'').replace(/[.+^$[\]\\(){}|-]/g, "\\$&");
}
function escapeRegExp(str) {
	return (str+'').replace(/[.+?*^$[\]\\(){}|-]/g, "\\$&");
}
function isBlank(str) {
	return (!str || /^\s*$/.test(str));
}
function replaceAllIgnoreCase(string, find, replace) {
	return string.replace(new RegExp(escapeRegExp(find), 'gi'), replace);
}

/* New functions, Eurlex 2.0 */

/*Method for autocomplete results with parameters:
 * inputField: jQuery element where autocomplete will be applied
 * url: application internal url
 * isQuickSearch: If the autocomplete results should be displayed for the Quick search field
 * divClass: the class of the div that contains the input field for which the autocomplete results should be displayed
*/
function typeaheadFld(inputField, url, isQuickSearch, divClass){
	var inputId = inputField.attr('id');
	//When clicking away of the suggestions list, remove any active class
	$('body').click(function(evt){
		//When target of click is inputField, or its list of suggestions	
		if ( (evt.target.id ==  inputId || $(evt.target).closest('.typeahead.dropdown-menu').siblings('#' + inputId).length) ) {
		} else {
			$('#' + inputId).siblings('.typeahead.dropdown-menu').find('li').removeClass('active');
		}
	});

	var typingTimer;
    var typingInterval = autocompleteTimer; // Autocomplete delay in milliseconds, configuration BO -> Website configutarion/Search preferences/Autocomplete Timer
    
	inputField.typeahead({
		source: function (query, process) {	//This callback does the actual submission of query and processes results.

        clearTimeout(typingTimer);
        typingTimer = setTimeout(function() {
            if (inputField.val().length > 2) {
                return $.get(url +'?searchString=' + encodeURIComponent(query) + '&quickSearch='+ isQuickSearch, function (data) {

                    if (data && data.csrf) {
                        //For performance, we set the new token value on all other elements only if the first csrfToken we find is expired (different than the one sent)
                        if ($('input[name="_csrf"]').length > 0 && $('input[name="_csrf"]')[0].value !== data.csrf) {
                            $('input[name="_csrf"]').attr('value', data.csrf);
                        }

                    }
                    //If we got results for autocomplete
                    if (data && data.suggestions && data.suggestions.length > 0) {

                        var newData = [];

                        $.each(data.suggestions, function( index, value ) {
                            // Sanitize html input to avoid XSS
                            var tmpDiv = document.createElement('div')

                            const sanitizedValue = value.value.replace(/[\r\n]/g, '');

                            tmpDiv.textContent = sanitizedValue;
                            newData.push($('<div/>').html(tmpDiv.innerHTML).text());
                        });

                        var result = process(newData);

                        //for each suggestion
                        $('.typeahead li a.dropdown-item').each(function(index, item) {
                            var $this = $(this);
                            var textareaElement = '$(this).closest(".typeahead").siblings("textarea, input")';
                            if (isQuickSearch) {	//When clicked, suggestions should replace the original text.
                                //escape the text in order to be able to assign to the DOM. Then unescape & submit.
                                $this.attr('onClick', textareaElement + '.val("' + escape(item.text) + '"); ' + textareaElement + '.val(unescape(' + textareaElement + '.val())); $(this).closest("form").submit();');
                                $this.attr('ontouchstart', textareaElement + '.val("' + escape(item.text) + '"); ' + textareaElement + '.val(unescape(' + textareaElement + '.val())); $(this).closest("form").submit();');
                            } else {	//When not in QS we should also know that form is ready for submission via ENTER, when we click select a value.
                                //escape the text in order to be able to assign to the DOM. Then unescape & add sReady.
                                $this.attr('onClick', textareaElement + '.val("' + escape(item.text) + '"); ' + textareaElement + '.val(unescape(' + textareaElement + '.val())); $(this).closest("form").addClass("sReady");');
                                $this.attr('ontouchstart', textareaElement + '.val("' + escape(item.text) + '"); ' + textareaElement + '.val(unescape(' + textareaElement + '.val())); $(this).closest("form").addClass("sReady");');
                            }
                        });

                        //these event handlers make hovered suggestions appear 'active' while not actually beeing. This avoids submission of hovered values.
                        var links = $("ul.typeahead > li");
                        links.mousemove(function(evt){
                            $(this).removeClass('active');
                        });
                        links.mouseenter(function(evt){
                            $(this).addClass('appearActive');
                        });
                        links.mouseleave(function(evt){
                            $(this).removeClass('appearActive');
                        });

                        return result;

                    } else {	// no results
                        //clear list options and hide
                        $('div.'+ divClass).find('ul.typeahead').hide();
                        $('div.'+ divClass).find('ul.typeahead').find('li').remove()
                    }
                });
            } else {	//no new search
                //clear list options and hide
                $('div.'+ divClass).find('ul.typeahead').hide();
                $('div.'+ divClass).find('ul.typeahead').find('li').remove();
                //show QS helper
                if(isQuickSearch) {
                    $('div.QuickSearchOptions').fadeIn('medium').removeClass('sr-only').addClass('in');
                }
            }
            }, typingInterval);
		},
		items:999, //max number of results is configured on server side, show as many as returned
		autoSelect: false,
		updater: function(item) {
			//When moving in suggestions retain the same value
			return this.$element[0].value;
		},
		matcher: function(item) {
			//Show suggestions whether they contain the term or not
			return true;
		},
		afterSelect: function(item) {	//Callback used after plugin 'selection'. Here we do not use it as we handle selections and submissions manually.
			return null;
		}
	}).on('keyup', this, function (event) {
		//when hitting ENTER key
		if (event.keyCode == 13) {
			if (isQuickSearch) {	//When in QS and ENTER is hit, we select possible highlighted element & submit always.
				if ($('.typeahead.dropdown-menu li.active').text().length > 0) {
					$(this)[0].value = $('.typeahead.dropdown-menu li.active').text();
				}
				showHourglass();
				$(this).closest('form')[0].submit();
			} else {	//When not in QS
				//If sReady class is there, we know that form is ready for submission via ENTER, so we should submit.
				if (($(this).closest('form').hasClass('sReady'))) {
					$(this).closest('form')[0].submit();
					showHourglass();
				} else {	//If sReady class is not there
					if ($('.typeahead.dropdown-menu li.active').text().length > 0) {	//If we have active suggestion
						//Select it and append sReady to submit on next ENTER.
						$(this)[0].value = $('.typeahead.dropdown-menu li.active').text();
						$(this).closest('form').addClass('sReady');
					} else {	//no active suggestion - submit original text.
						$(this).closest('form')[0].submit();
						showHourglass();
					}
				}
			}
		} else { //other keys
			if (!isQuickSearch) {	//Changed input: remove sReady class
				$(this).closest('form').removeClass('sReady');
			}
		}
	});
}

function checkPagingFO(el, nbreOfPage) {
	try {
		if($(el).val() == '') {
			$(el).addClass('NoBorders');
			$(el).removeClass('pagingInvalid');
			return;
		}
		var currentPage = parseInt($(el).val());
		if(parseFloat($(el).val()) != currentPage || isNaN(currentPage) || endWith($(el).val(), '.')) {
			$(el).addClass('pagingInvalid');
			$(el).removeClass('NoBorders');
			return;
		}
		var maxPage = parseInt(nbreOfPage);
		if(currentPage > maxPage || currentPage == 0) {
			$(el).removeClass('NoBorders');
			$(el).addClass('pagingInvalid');
		} else {
			$(el).addClass('NoBorders');
			$(el).removeClass('pagingInvalid');
		}
	} catch (e) {
		$(el).removeClass('NoBorders');
		$(el).addClass('pagingInvalid');
	}
}

function submitPagingFO(el, nbreOfPage) {
	try {
		if($(el).val() == '') {
			$(el).addClass('NoBorders');
			$(el).removeClass('pagingInvalid');
			return;
		}
		var currentPage = parseInt($(el).val());
		if(parseFloat($(el).val()) != currentPage || isNaN(currentPage) || endWith($(el).val(), '.')) {
			$(el).removeClass('NoBorders');
			$(el).addClass('pagingInvalid');
			return false;
		}
		var maxPage = parseInt(nbreOfPage);
		if(currentPage > maxPage || currentPage == 0) {
			$(el).removeClass('NoBorders');
			$(el).addClass('pagingInvalid');
			return false
		} else {
			$(el).addClass('NoBorders');
			$(el).removeClass('pagingInvalid');
			return true;
		}
	} catch (e) {
		$(el).removeClass('NoBorders');
		$(el).addClass('pagingInvalid');
		return false;
	}
}

function createHpCookie(clickedPanel){
	//If we clicked an already expanded panel. Search panels are mutually exclusive so if we clicked an already expanded one, cookie should remember that both are closed.
	if ( ($('#QSByDocNumber').hasClass('in') && clickedPanel == 'documentNumberSearch') || ($('#QSByCelex').hasClass('in') && clickedPanel == 'celexSearch') ) {
		createCookie('stab', 'none', 30);
	} else {
		if (clickedPanel == 'celexSearch') {
			createCookie('stab', 'celexSearch', 30);
		} else { //documentNumberSearch
			createCookie('stab', 'documentNumberSearch', 30);
		}
	}
}

//Async. Text load for Notice pages when in mobile.
function loadDocText(noticeUrl, queryString, internalUrl, topLabel, errorLabel){
	//remove button
	$("#text #textLoadBtn").remove();

	$("#text").append('<i id="loadingImg" class="fa fa-spinner fa-spin" aria-hidden="true"></i>');
	$("#text").addClass('text-center');

	if (queryString.length == 0){
		return false;
	}
	var completeUri = internalUrl.substring(0, internalUrl.length - 1) + noticeUrl + "?" + queryString;

	//Async. Ajax call to server to get text.
	var myJSONTimeout;
	clearTimeout(myJSONTimeout);
	myJSONTimeout = window.setTimeout(function(){
		$.getJSON({
			url:completeUri,
			cache: false,
			data: {ajaxLoadText: true},
			success: function(response){
				//parse the JSON object
				//Outer loop: removes wrapper
				var indx = 0;
				$.each(response.values, function( index, value ) {
					//Inner loop: for each doc. stream
					$.each(value, function( str, content ) {
						if (indx > 0 ) {
							$('#textTabContent').append('<div class="documentSeparator tabContent"><br/></div>');
						}
						//Do wrappings as in normal jsp display and append content
						var wrappingBefore = '<div id="document'+ (index + 1) + '" class="tabContent">' + '<div class="tabContent">' + '<div lang="'+ response.lang +'">';
						var wrappingAfter = '</div>' + '<a href="#document' + (index + 1) + '">' + topLabel + '</a>' + '</div></div>';

						$('.ojDisabled').removeClass('ojDisabled');
						content = content.replace('./../../../../../', './../../../../');
						$('#textTabContent').append(wrappingBefore + content + wrappingAfter);
						$('#ojTabContent').append(wrappingBefore + content + wrappingAfter);
						indx = indx + 1;
					});
				});

				$("#text #loadingImg").remove();
				$("#text").removeClass('text-center');

				// Check if the document's text has been loaded.
				if ($("#document1").length !== 0) {
				    // Make TOC buttons available.
				    $("#tocBtn").removeClass('hidden'); // PC/Tablet.
				    $("#tocBtnMbl").removeClass('hidden'); // Phone.

				    // Trigger an update of the TOC layout.
				    $("#documentView").trigger("eurlex:toc.layout.update");
				}
			},
			error: function(response){
				$("#text #loadingImg").remove();
				$("#text").removeClass('text-center');
				$("#text").append("<div class='alert alert-danger' role='alert'><span class='fa fa-exclamation-circle' aria-hidden='true'>&nbsp;</span>"
				        + errorLabel + "</div>");
			},
		});
	}, 500);
}


function generatePdfComponent(uniqueId,pdfUrl,internalUrl) {

		var div_iframe = $('#div-iframe');

		//remove button
		$("#text  #btn-generate-pdf").remove();

		$("#text").append('<i id="loadingImg" class="fa fa-spinner fa-spin" aria-hidden="true"></i>');
		$("#text").addClass('text-center');


		const infiniteList = `<div id="infinite-list-${uniqueId}" class="infinite-list"></div>`;
		const contentInfo = `<div id="content-info-${uniqueId}" class="content-info"></div>`;



	$(document).ready(function(){

		$("#text").removeClass('text-center');
		$("#text #loadingImg").remove();
		$('#div-iframe').append(`<div>${infiniteList}${contentInfo}</div>`);
		pdfComponent(uniqueId, pdfUrl,internalUrl);
	});
}

//Check if a checkbox is selected in pages My-Eurlex and search-preferences
function isCheckboxSelected(checkboxId) {
	var isSelected = false;
	$("input[id^='"+checkboxId+"']").each( function(){
		if (this.checked){
			isSelected = true;
			return false;
		}
	});
	return isSelected;
}

//Search results sort order: Updates the arrow displayed on select, so as to not wait for page load.
function updateSortingArrow(el, val) {
	var arrowButton = $(el).closest('ul').siblings('button');
	arrowButton.find('i').each(function( index ) {
		if (index == 0){
			if (val == 'asc') {
				$(this).removeClass();
				$(this).addClass('fa fa-long-arrow-up');
			} else {
				$(this).removeClass();
				$(this).addClass('fa fa-long-arrow-down');
			}
		} else {
			if (val == 'asc') {
				$(this).removeClass();
				$(this).addClass('fa fa-angle-up');
			} else {
				$(this).removeClass();
				$(this).addClass('fa fa-angle-down');
			}
		}
	});
}

// EURLEXNEW-3625: 'Collapse all'/'Expand all' for revamped procedure view.
function toggleAllProc(collapse) {
    const cdTimelineBlock = $(".cd-timeline__block");
    const instDetails = cdTimelineBlock.find(".instDetails");
    const instDetailsCollapse = instDetails.find(".collapse");

    instDetails.map(function(_, obj) {
        createCookie("Procedure" + obj.id, (collapse ? '0' : '1'), 30);
    });
    instDetailsCollapse.map(function(_, obj) {
        createCookie("Procedure" + obj.id, (collapse ? '0' : '1'), 30);
    });

    instDetails.collapse(collapse ? "hide" : "show");
    instDetailsCollapse.collapse(collapse ? "hide" : "show");

    cdTimelineBlock.find(".ViewMoreInfo > i.fa")
                   .removeClass(collapse ? "fa-minus-square" : "fa-plus-square")
                   .addClass(collapse ? "fa-plus-square" : "fa-minus-square");
}

function expandAllProc()   { toggleAllProc(false); }
function collapseAllProc() { toggleAllProc(true);  }

// EURLEX-4891: 'Collapse/Expand all' and download csv/xml for BOL.
function toggleAllBol(collapse) {
    const cdBolBlock = $(".cd-bol-group");
    const instDetails = cdBolBlock.find(".instDetails");
    const instDetailsCollapse = instDetails.find(".collapse");

    instDetails.collapse(collapse ? "hide" : "show");
    instDetailsCollapse.collapse(collapse ? "hide" : "show");

    cdBolBlock.find(".ViewMoreInfo > i.fa")
                   .removeClass(collapse ? "fa-minus-square" : "fa-plus-square")
                   .addClass(collapse ? "fa-plus-square" : "fa-minus-square");
}

function expandAllBol()   { toggleAllBol(false); }
function collapseAllBol() { toggleAllBol(true); }

function downloadCsvBol() { return; }
function downloadXmlBol() { return; }

// EURLEXNEW-3644: Cookies for revamped procedure view.
function createDocPartCookieInst(el) {	// Institution level.
    const elementID = $(el).closest(".cd-timeline_title").siblings(".instDetails").attr("id");
    createCookie("Procedure" + elementID, ($(el).hasClass('collapsed') ? '1' : '0'), 30);
}

function createDocPartCookieEvt(el) {	// Event level.
	const controlledDivId = $(el).attr("aria-controls");
	createCookie("Procedure" + controlledDivId, ($("#" + controlledDivId).hasClass('in') ? '0' : '1'), 30);
}

/* EURLEXNEW-3662: E-Learning Content - Transfer to Normal Editorial Pages */
/**
 *  Initializes quizzes and their functionality in 'E-Learning' editorial pages.
 */
function ELQuizzes() {
    // Set a handler to be executed when a quiz is submitted.
    $('button.quizSubmit').click(function() {
        const self = $(this);

        const quizForm = self.closest('form');
        const quizFormID = quizForm.attr('id');

        // Check if the current quiz is a 'simple' quiz. (one correct answer)
        if (quizForm.hasClass('quizSimple')) {
            const selection = quizForm.find('input:checked');

            if (selection.length === 1) {
                // Save most recent selection.
                const selectionID = selection.attr('id');
                sessionStorage.setItem(quizFormID, selectionID);

                // Update quiz history: store all (distinct) selections ever
                // submitted.
                const quizHistoryKey = quizFormID + 'history';
                let quizHistory =
                        sessionStorage.getItem(quizHistoryKey, selectionID);
                if (quizHistory == null) {
                    sessionStorage.setItem(quizHistoryKey, selectionID);
                } else {
                    quizHistory = sanitizeHtml(quizHistory);
                    if (quizHistory.indexOf(selectionID) === -1) {
                        sessionStorage.setItem(
                                quizHistoryKey, quizHistory + ',' + selectionID);
                    }
                }

                // Toggle quiz annotations based on if the selected answer is correct.
                if (selection.hasClass('rightAnswer')) {
                    quizForm.find('input').siblings('small').removeClass('hidden');

                    // 'Skip quiz' becomes 'Continue'.
                    quizForm.find('.pull-right a').addClass('hidden');
                    quizForm.find('#quizSimpleContinue').removeClass('hidden');
                } else {
                    selection.siblings('small').removeClass('hidden');
                }
            }

            // Display all quiz annotations, if only one is left to display.
            if (quizForm.find('small.hidden').length === 1) {
                quizForm.find('input').siblings('small').removeClass('hidden');

                // 'Skip quiz' becomes 'Continue'.
                quizForm.find('.pull-right a').addClass('hidden');
                quizForm.find('#quizSimpleContinue').removeClass('hidden');
            }
        }

        // Check if the current quiz is a 'multiple-choice' quiz. (multiple
        // correct answers)
        if (quizForm.hasClass('quizMulti')) {
            const selections = quizForm.find('input:checked');

            if (selections.length > 0) {
                // Save all current selections.
                let selectionIDs = [];
                selections.each(function() {
                    selectionIDs.push($(this).attr('id'));
                });
                sessionStorage.setItem(quizFormID, selectionIDs);

                // Display all quiz annotations.
                quizForm.find('input').siblings('small').removeClass('hidden');

                // 'Skip quiz' becomes 'Continue'.
                quizForm.find('.pull-right a').addClass('hidden');
                quizForm.find('#quizMultiContinue').removeClass('hidden');
            }
        }
    });

    // Initialize the functionality of all quizzes present in the current page.
    $('.ELQuiz').each(function() {
        const quizForm = $(this);
        const quizFormID = quizForm.attr('id');

        // Check if the current quiz is a 'simple' quiz. (one correct answer)
        if (quizForm.hasClass('quizSimple')) {
            // Retrieve the previous answer (if any), and re-submit it.
            const previousAnswer = sessionStorage.getItem(quizFormID);
            if (previousAnswer != null) {
                quizForm.find('#' + sanitizeHtml(previousAnswer))
                        .prop('checked', true);
                quizForm.find('.quizSubmit').click();
            }

            // Retrieve all answers historically submitted, and display their
            // annotations.
            const quizHistory = sessionStorage.getItem(quizFormID + 'history');
            if (quizHistory != null) {
                const answers = sanitizeHtml(quizHistory).split(',');
                answers.forEach(function(answer) {
                    answer = quizForm.find('#' + answer);
                    answer.siblings('small').removeClass('hidden');

                    const inputs = quizForm.find('input');
                    if (answer.hasClass('rightAnswer') ||
                        (answers.length >= inputs.length - 1)) {
                        inputs.siblings('small').removeClass('hidden');

                        // 'Skip quiz' becomes 'Continue'.
                        quizForm.find('.pull-right a').addClass('hidden');
                        quizForm.find('#quizSimpleContinue').removeClass('hidden');
                    }
                });
            }
        }

        // Check if the current quiz is a 'multiple-choice' quiz. (multiple correct
        // answers)
        if (quizForm.hasClass('quizMulti')) {
            // Retrieve all previous answers (if any), and re-submit them.
            const previousAnswers = sessionStorage.getItem(quizFormID);
            if (previousAnswers != null) {
                sanitizeHtml(previousAnswers).split(',').forEach(function(answer) {
                    quizForm.find('#' + answer).prop('checked', true);
                    quizForm.find('.quizSubmit').click();
                });
            }
        }
    });
}

//This function is called on doc. load, and initializes 'main features' behavior in EL pages
function ELMainFeatures() {
	//Call mapster plugin on our image
	var img = $("#main_features");
	img.mapster(
		{
			isSelectable: true,
			singleSelect: false,	//multiple selections (number + area)
			mapKey: "id",
			listKey: "id",
			render_highlight: {
				altImage: img.attr("src"),	//use original image for highlighted area
				fillColor: "transparent",
				fillOpacity: 1
			},
			render_select: {
				altImage: img.attr("src"),	//use original image for selected area
				fillColor: "transparent",
				fillOpacity: 1
			}
		});

	//Wait for ImageMapster to append its classes, then call and bind resizeELMF method
	var checkExist = setInterval(function() {
		if ($(".mapster_el").length) {
			clearInterval(checkExist);
			resizeELMF(img);
			$( window ).resize(function() {
				resizeELMF(img);
			});
		}
	}, 100); // check every 100ms
}

//This function is used to adjust 'main features' image map based on available space
function resizeELMF(img) {
	//If some tooltip was open (between resizes), close it and deselect
	if ($(".ELTooltip").length > 0) {
		$(".ELTooltip").remove();
		$("area").mapster("deselect");
	}
	img.removeClass("hidden");	//only useful the first time, image is hidden to avoid flickering
	//unbind any existing mapster from past resolutions (between resizes)
	img.mapster("unbind");

	img.mapster({
		isSelectable: true,
		singleSelect: false,	//multiple selections (number + area)
		mapKey: "id",
		listKey: "id",
		render_highlight: {
			altImage: img.attr("src"),	//use original image for highlighted area
			fillColor: "transparent",
			fillOpacity: 1
		},
		render_select: {
			altImage: img.attr("src"),	//use original image for selected area
			fillColor: "transparent",
			fillOpacity: 1
		}
	});

	img.mapster("resize", $("#MF_container").innerWidth(), 0, 0);		//resize based on container width (available space)
	var whiteFilter = $("<div />").width($("#MF_container").innerWidth()).height(img.height())	//define our white filter based on container width (available space)
		.attr("id","white-filter")
		.css({
			position: "absolute",
			left: "0",
			top: "0",
			backgroundColor: "white",
			opacity: 0.5
		});

	$(".MF_number").unbind();	//Unbind all handlers from past resolutions (between resizes)
	$(".MF_area").unbind();

	//MF_number area handlers
	$(".MF_number").on("mouseover", function() {	//when hovering over number area, higlight both number & ref. areas, also apply white filter
		img.parent().find(".mapster_el").eq(0).after(whiteFilter);
		$(this).mapster("highlight");
		$(this).next(".MF_area").mapster("highlight");
	}).on("mouseout", function() {	//when done hovering over number area, 
		whiteFilter.remove();	//remove white filter
	}).on("click", function() {	//when clicking number area, select both number & ref. areas and display a tooltip
		$("area").mapster("deselect");
		$(this).mapster("select");
		$(this).next(".MF_area").mapster("select");

		if ($(".ELTooltip").length == 0) {
			createELTooltip($(this));
		} else {
			$(".ELTooltip").remove();	//one tooltip at a time
			createELTooltip($(this));
		}
	});

	//MF_area area handlers
	$(".MF_area").on("mouseover",function(e) {	//when hovering over ref. area, higlight nothing
		img.mapster("highlight", false);
	});
}

//creates the tooltip for a selected number + reference area pair
function createELTooltip(el) {
	//Get coordinates of number area and calculate tooltip position
	var coords = el.attr("coords");
	var coordArray = coords.split(',');
	var topC = parseInt(coordArray[1]) + parseInt(coordArray[2]);
	var leftC = parseInt(coordArray[0]);
	var style = "top:" + topC + "px;left:" + leftC + "px;display:none;";
	//Get tooltip contents from DOM
	var title = $("#" + el.attr("id") + "_title").html();
	var content = $("#" + el.attr("id") + "_body").html();
	$("#mapster_wrap_0").append(
	        "<div class='ELTooltip' style=" + style + "><div class='popover-title'>" + title
	        + "<i class='fa fa-close' aria-hidden='true'></i></div><div class='popover-content'>"
	        + content + "<div></div>");

	//appended width would be calculated via css only
	var initWidth = $(".ELTooltip").innerWidth()/2;
	//calculate new position based on available screen size
	if (leftC < initWidth) {
		$(".ELTooltip").css("left", 0);
	} else if ($("#main_features").width() - leftC < initWidth) {
		$(".ELTooltip").css("left", "");
		$(".ELTooltip").css("right", 0);
	} else {
		$(".ELTooltip").css("left", parseInt($(".ELTooltip").css("left")) - $(".ELTooltip").innerWidth()/2);
	}
	//show tooltip
	$(".ELTooltip").css("display","block");

	//Bind this onclick of X button, close tooltip & deselect
	$(".ELTooltip .fa-close").on("click", function () {
		$(this).closest(".ELTooltip").remove();
		$("area").mapster("deselect");
	});
}

/**
 *  Handles the 'visited e-Learning menu links' state, and adds
 *  bullets to visited e-Learning menu links accordingly.
 *
 *  @param  requestURL  The current e-Learning page URL.
 */
function ELMenu(requestURL) {
    // Apply HTML sanitization to the 'requestURL' parameter for
    // security reasons.
    requestURL = sanitizeHtml(requestURL);

    const menuLinks = $('.eLearningMenuList li ul li a');
    menuLinks.unbind(); // Remove all previous event handlers.

    // Retrieve the current 'visited e-Learning menu links' state.
    const visitedLinks = sessionStorage.getItem('visitedElItems');
    if (visitedLinks != null) {
        const currentPage = /[^/]*$/.exec(requestURL)[0];

        sanitizeHtml(visitedLinks).split(',')
                                  .forEach(function(visitedLink) {
            // Retrieve the current 'visited' link from session storage.
            let link = /[^/]*$/.exec(visitedLink)[0];

            if (link !== currentPage) {
                // Create & append a 'bullet' icon to the menu link item.
                link = menuLinks.filter("[href*='" + visitedLink + "']");
                const bullet = $(document.createElement('i'))
                        .addClass('fa fa-circle')
                        .attr('aria-hidden', 'true');
                link.append(bullet);
            }
        });
    }

    // Create a handler that will store the current menu link, if it
    // is not already marked as 'visited'.
    const storeCurrentLink = function() {
        const currentLink = $(this).attr('href');

        // Retrieve the current 'visited e-Learning menu links' state.
        let visitedLinks = sessionStorage.getItem('visitedElItems');
        if (visitedLinks == null) {
            sessionStorage.setItem('visitedElItems', currentLink);
        } else {
            visitedLinks = sanitizeHtml(visitedLinks);

            // Check if the current menu link has not been saved before.
            if (visitedLinks.indexOf(currentLink) === -1) {
                visitedLinks += ',' + currentLink;
                sessionStorage.setItem('visitedElItems', visitedLinks);
            }
        }
    };
    // Add the handler to menu items, to store the current menu link on
    // 'click'.
    menuLinks.click(storeCurrentLink);
    $('.ELQuiz .linkAsBtn').click(storeCurrentLink);
}

//This function is called on doc. load, and initializes 'Collections' behavior in EL pages
function ELCollections() {
	$(".collectionTabs li a").click(function() {	//when tab is clicked
		//deselect all bubbles, and hide info
		$(".collectionsList li").removeClass("active");
		$("#collectionInfo").addClass("hidden");

		var thisId = $(this).attr("id");
		var classSelected = thisId.substr(0, thisId.indexOf('_tab'));
		//remove highlight of previous category
		$(".collectionsList li").removeClass("selected");
		//highlight current category
		$("." + classSelected).addClass("selected");
	});
	$(".collectionsList li").click(function() {	//when bubble is clicked
		//click the corresponding category tab, to highlight it
		$("#" + $(this).attr("class") + "_tab").click();
		//deselect all bubbles, select current
		$(".collectionsList li").removeClass("active");
		$(this).addClass("active");
		//append corresponding translation from DOM to info and show it
		$("#collectionInfo").empty();
		var targetHtml = $("#" + $(this).attr("id") + "_info").html();
		$("#collectionInfo").append(targetHtml)
		$("#collectionInfo").removeClass("hidden");
	});
}

/* EURLEXNEW-3745 (start) */
function initConsLegTable(screenWarn) {
	$("#showConsLegVersions").click(function() {
	    // Check if the document's text has been loaded.
        if ($("#document1").length !== 0) {
            $("#tocHideBtn").click(); // Hide the TOC panel.
        }

		$("#showConsLegVersions").addClass("hidden");
		$("#hideConsLegVersions").removeClass("hidden");
		$(".consLegNav").removeClass("hidden");
		adjustConslegHeight(screenWarn);
		//scroll to open version
		$(".consLegNav").animate({
			scrollTop: $(".consLegNav .active").offset().top - $(".consLegNav").offset().top + $(".consLegNav").scrollTop()
		});
	});
	$("#hideConsLegVersions").click(function() {
		$("#showConsLegVersions").removeClass("hidden");
		$("#hideConsLegVersions").addClass("hidden");
		$(".consLegNav").addClass("hidden");
		$("#tocSidebar").attr("style", "");
		if (!($("#consLegVersions .alert-info").length == 0) && !$("#consLegVersions .alert-info").hasClass("hidden")) {
			$("#consLegVersions .alert-info").addClass("hidden");
		}
	});

	//Append current and active classes to the versions table by retrieving  info from hidden divs
	$("#currentConsLeg").html();
	$(".consLegNav a").each(function() {
		if ($(this).html() == $("#currentConsLeg").html()) {
			$(this).addClass("current");
		}
		if ($(this).html() == $("#activeConsLeg").html()) {
			$(this).addClass("active");
		}
	});
	$("#legalActLink").attr("href", $(".consLegLinks > a").attr("href"));

    const affixSidebar = $("#AffixSidebar");
	const documentView = $("#documentView");

	const consLegVersions = $("#consLegVersions");
	const consLegVersionsNAV = consLegVersions.find(".consLegNav");
	const showConsLegVersions = $("#showConsLegVersions");
	const hideConsLegVersions = $("#hideConsLegVersions");

	const getElementActualHeight = function(element) {
        return ((element.length !== 0) && !element.hasClass("hidden"))
                ? element.height()
                : 0; // Zero height for non-existent or invisible elements.
    };

	const recalculateDocumentViewLayout = function() {
	    // Create the necessary space under the document view container,
        // in order for the left menu's components not to fall into the
        // page's footer.
        let marginBottom = affixSidebar.height()
                + parseInt(affixSidebar.css("margin-bottom"))
                + parseInt(consLegVersions.css("margin-top"))
                + getElementActualHeight(consLegVersions.find(".alert"))
                + getElementActualHeight(showConsLegVersions)
                + getElementActualHeight(hideConsLegVersions)
                + getElementActualHeight(consLegVersionsNAV)
                - documentView.height()
                + 4; // Extra offset to avoid glitches if possible.

        // Check if the 'Consolidated versions' panel is hidden (closed),
        // and the space between the 'Consolidated versions' button and
        // the page's footer is very small. (The button is near the footer)
        if (hideConsLegVersions.hasClass("hidden") && (marginBottom > 0)) {
            // Add an extra offset (in pixels) to the bottom margin of the
            // document view, so that the user is not blocked from opening
            // the 'Consolidated versions' panel later.
            // It is recommended that this value is at least 90 pixels.
            marginBottom += 100;
        }

        documentView.css("margin-bottom", (marginBottom > 0)
                ? (marginBottom + "px")
                : ""); // Clear the bottom margin on negative/zero value.
    };
    showConsLegVersions.click(recalculateDocumentViewLayout);
    hideConsLegVersions.click(recalculateDocumentViewLayout);

    // Called by other components (e.g. TOC) to update the layout of the document
    // view container when necessary.
    documentView.on("eurlex:consleg.layout.update", recalculateDocumentViewLayout);

	const recalculateConsLegLayout = function(calculateHeight) {
        if (calculateHeight) calculateConslegHeight();
        adjustConslegHeight(screenWarn);
    };
    recalculateConsLegLayout(true);

    const documentTextPanel = $("#PP4Contents");
	const contentPanels = $(".panel-collapse.collapse");
    if (contentPanels.length > 0) {
        // Observes 'class' attribute changes on content panels.
        const contentPanelObserver = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.attributeName === "class") {
                    // Adjust the 'Consolidated versions' button when content
                    // panels are in the 'expanded'/'collapsed' state.
                    recalculateConsLegLayout(false);

                    // Re-Calculate the layout of the document view container.
                    recalculateDocumentViewLayout();
                }
            });
        });

        contentPanels.each(function() {
            contentPanelObserver.observe(this, { attributes: true });
        });
    }

    affixSidebar.on('affixed.bs.affix', function() {
		recalculateConsLegLayout(false);
	}).on('affixed-top.bs.affix', function() {
		recalculateConsLegLayout(false);
	}).on('affixed-bottom.bs.affix', function() {
		recalculateConsLegLayout(false);
	});

	$(window).on("load resize", function() {
        recalculateConsLegLayout(true);
	}).on("scroll", function() {
		recalculateConsLegLayout(false);
	});
}
/* EURLEXNEW-3745 (end) */

/* EURLEXNEW-4360 */
function calculateConslegHeight() {
    const availableNavHeight = $(window).height() - $("#AffixSidebar").height() - 95;
	$("#consLegVersions .consLegNav").css("max-height", availableNavHeight);
}

function adjustConslegHeight(screenWarn) {
    const affixSidebar       = $("#AffixSidebar");
    const consLegVersions    = $("#consLegVersions");
    const consLegVersionsNAV = consLegVersions.find(".consLegNav");

    const isAffixTop     = affixSidebar.hasClass("affix-top");
    const maxHeight      = parseInt(consLegVersionsNAV.css("max-height"));
	const isFooterInView = calculateViewportFooterDifference() > 0;

	let isOverFlown, availableNavHeight = "";

	if (isFooterInView) {
		availableNavHeight = $("footer").offset().top - consLegVersionsNAV.offset().top - 10;
	} else if (isAffixTop) {
		availableNavHeight = maxHeight - affixSidebar.offset().top + $(window).scrollTop() + 120;
	}

	if (availableNavHeight === "") { // Middle of the page.
	    // Clear 'height' CSS property so 'max-height' applies.
		consLegVersionsNAV.css("height", "");

		// If space is enough, remove any previously appended warning message.
		if (maxHeight > 0) {
			consLegVersions.find(".alert-info").addClass("hidden");
		}
	} else { // Top/Bottom of the page.
	    // Check whether the 'Consolidated versions' panel is overflown if it is expanded,
	    // otherwise retain the previous value.
	    if (!consLegVersionsNAV.hasClass("hidden")) {
	        isOverFlown = consLegVersionsNAV.prop("scrollHeight") > availableNavHeight;
	    }

		if (isAffixTop && maxHeight > 0) {
		    // Remove any previously appended warning message.
		    consLegVersions.find(".alert-info").addClass("hidden");

            // Clear 'height' CSS property so 'max-height' applies.
            consLegVersionsNAV.css("height", "");
		}

		if (availableNavHeight < maxHeight && isOverFlown && !(isAffixTop && !isFooterInView)) {
			availableNavHeight = checkRemainingSpace(availableNavHeight, screenWarn);
			consLegVersionsNAV.css("height", availableNavHeight);
        }
	}

	/*
     *  EURLEXNEW-4654: Slide the left sidebar up to avoid the 'Consolidated versions' panel
     *  shrinking and (possibly) hiding its contents, because of the available space being too
     *  small.
     */
	adjustLeftSidebarTopPosition(isAffixTop ? "" : "5px");
}

function checkRemainingSpace(availableNavHeight,screenWarn) {
	if (availableNavHeight < 60) {	//Limited space
		availableNavHeight = 0;
	}

	//true on first compute, false on resizing
	var justCalculated = !$("#consLegVersions .consLegNav").hasClass("hidden");
	//Logic to add warning msg
	if (availableNavHeight == 0) {
		if ($("#consLegVersions .alert-info").length == 0) {	//if warning not yet in DOM add it
			$("#consLegVersions").prepend(
			        "<span class='alert alert-info' role='alert'><i class='fa fa-exclamation' aria-hidden='true'>&nbsp;</i>"
			        + screenWarn + "</span>");
		} else {	//just show it
			$("#consLegVersions .alert-info").removeClass("hidden");
		}
	} else {
		$("#consLegVersions .alert-info").addClass("hidden");	//hide  any previously appended warning
	}

	return availableNavHeight;
}

/*
 *  EURLEXNEW-4654: Slide the left sidebar up to avoid (possible) collisions
 *  between left sidebar elements and the footer of the page, when the page's
 *  footer starts getting into view.
 */

// This variable will hold the initial difference (in pixels) between the
// viewport and the page's footer, before the left sidebar starts sliding up.
// This value is needed to correct the slide offset, as there is a special
// case where the page's footer may become visible before the left sidebar
// starts sliding up.
let initialViewportFooterDifference = 0;

/*
 *  Adjusts the left sidebar's top position, to avoid (possible) collisions
 *  between left sidebar elements and the footer of the page, when the page's
 *  footer starts getting into view.
 *
 *  @param  defaultTopValue  The default 'top' CSS property value to use when
 *                           the left sidebar's position should be restored.
 */
function adjustLeftSidebarTopPosition(defaultTopValue) {
    const affixSidebar = $("#AffixSidebar");
    const documentTextPanel = $("#PP4Contents");

	const isAffixTop = affixSidebar.hasClass("affix-top");

	let viewportFooterDifference = calculateViewportFooterDifference();
	if (viewportFooterDifference > 0) {
	    // Store the initial viewport-footer difference for offset value
	    // corrections later.
	    if (initialViewportFooterDifference === 0) {
	        initialViewportFooterDifference = viewportFooterDifference;

	        // Correct a visual glitch which appears if the page's footer
	        // is already visible, and the left sidebar is already at the
	        // top of the page.
	        if (isAffixTop) initialViewportFooterDifference += 168;
	    }

	    // Compute the final viewport-footer difference, taking also into
	    // account any initial offset. (only when the 'text' panel of the
	    // document is collapsed)
	    if (!documentTextPanel.hasClass("in"))
	        viewportFooterDifference -= initialViewportFooterDifference;

	    // Slide the left sidebar up to avoid a collision with the footer.
        affixSidebar.css("top", (4 - viewportFooterDifference) + "px");
    } else {
        // Reset the initial viewport-footer difference. (offset)
        initialViewportFooterDifference = 0;

        // Restore the top position of the left sidebar to the provided
        // default CSS value.
        affixSidebar.css("top", defaultTopValue);
    }
}

/**
 *  Calculates the distance between the viewport's bottom line, and
 *  the page footer's top part. (in pixels)
 *
 *  @return     The viewport-footer difference. A positive value
 *              indicates that the footer is currently visible (whether
 *              partially or as a whole).
 */
function calculateViewportFooterDifference() {
    const footerTop      = $("footer").offset().top;
    const viewportBottom = $(window).scrollTop() + $(window).height();

    return viewportBottom - footerTop;
}

/* EURLEXNEW-3967 (start) */
function highlightExpertQuery() {
	/*var textareaContainer = $(".ExpertSearch .ExpertSearchQuery .textareaContainer");
	var containerHeight = textareaContainer.height();
	$(".ExpertSearch .ExpertSearchQuery .backdrop").height(containerHeight);
	textareaContainer.css('margin-top', - containerHeight);*/
	$(".ExpertSearch .ExpertSearchQuery textarea#expertQuery").on('scroll', function() {
		var scrollTop = $(this).scrollTop();
		$(".ExpertSearch .ExpertSearchQuery .backdrop").scrollTop(scrollTop);
	})

	if($(".ExpertSearch .col-md-6 .alert-danger").length){
		var text = $(".ExpertSearch .col-md-6 .alert-danger").text();
		var lineRegex = /line (.*?) /g.exec(text);
		if (lineRegex.length > 1) {
			var line = lineRegex[1];
		}
		var charRegex = /character (.*?)\./g.exec(text);
		if (charRegex.length > 1) {
			var char = charRegex[1];
		}
		if (line && char) {
			highlightedQuery = getHighlightedQuery(line, char);
		}
		$(".ExpertSearch .ExpertSearchQuery .backdrop .highlights").html(highlightedQuery);
	}
}

function getHighlightedQuery(line, char) {
	var expertQuery = $(".ExpertSearch .ExpertSearchQuery textarea#expertQuery").val();
	var expertQuerySplit = expertQuery.split(/\r?\n/);
	var errorLine = expertQuerySplit[line-1];
	if (char == 0){
		errorLine = '<span class="highlight">' + errorLine.substring(0, errorLine.indexOf(' ')) + '</span>' + errorLine.substring(errorLine.indexOf(' '));
	} else if (char == errorLine.length) {
		errorLine = errorLine.substring(0, errorLine.lastIndexOf(' ')) + '<span class="highlight">' + errorLine.substring(errorLine.lastIndexOf(' ')) + '</span>';
	} else {
		errorLine = errorLine.substring(0, char) + '<span class="highlight">' + errorLine.substring(char) + '</span>';
	}
	expertQuerySplit[line-1] = errorLine;
	var highlightedQuery = expertQuerySplit.join("\r\n");
	return highlightedQuery;
}
/* EURLEXNEW-3967 (end) */

// EURLEXNEW-3278: Callback method for invoking CAPTCHA on-demand.
// Useful when the CAPTCHA is embedded into a modal.
function renderCaptcha() {
	$wt.render("captchaWrapper", {
		"service" : "captcha"
	});
}

// Function for dynamically retrieving translations in a specified language or for the current local
// The location (table/id) and optionally the required languae is specified in js-WTLabels.properties
// This function is generic and not restricted to WT usage.
function getWTLabel(labelId) {
	var label = (typeof(WTLabels) != 'undefined' && WTLabels) ? WTLabels[labelId] : null;
	if (!label) {
		console.warn("Missing WT label translation '" + labelId + "'.");
	}
	return label;
}

// Function for removing cookies related to WT.
function removeWebtrendsCookies() {
	if (readCookie("WT_FPC") != "") {
		deleteCookie("WT_FPC");
	}
	if (readCookie("WTLOPTOUT") != "") {
		deleteCookie("WTLOPTOUT");
	}
}

//EURLEXNEW-4131
function guidedTourTagging() {
	if ($('div.guidedTourTextDiv').length > 0) {
		initGuidedTourPlayer();
	}
}

//guidedTour related functions
function init_events(id, video) {
	video.addEventListener("play", play, false);
}

//guidedTour related functions
function initGuidedTourPlayer() {
	var video = document.getElementById('video');
	init_events("events", video);
}


function loadSessionItemList(item) {
	var itemId = '#'+$(item).attr('id');
	if($(itemId).hasClass('in')) {
		$(itemId).removeClass('in');
	} else {
		$(itemId).addClass('in');
	}
}

$(document).ready(function(){
    var all = $(".help-block").map(function() {
        return $(this).parent().find("input");
    }).get();
    $(all).each(function(){
        $(this).attr("aria-invalid","true");
    });
});

function copyClipboard(idForCopy){
    if (window.getSelection) {
        if (window.getSelection().empty) { // Chrome
            window.getSelection().empty();
        } else if (window.getSelection().removeAllRanges) { // Firefox
            window.getSelection().removeAllRanges();
        }
    } else if (document.selection) { // IE?
        document.selection.empty();
    }

    if (document.selection) {
        var range = document.body.createTextRange();
        range.moveToElementText(document.getElementById(idForCopy));
        range.select().createTextRange();
        document.execCommand("copy");
    } else if (window.getSelection) {
        var range = document.createRange();
        range.selectNode(document.getElementById(idForCopy));
        window.getSelection().addRange(range);
        document.execCommand("copy");
    }
}


//EURLEXNEW-4589 Summary pages: Methods to insert/remove css classes and rewriting the Export Pdf Url
//according to the expanded links.
function expandAllSummariesChapterNodes(summariesTree, initialExportPdfUrl, expandIconUrl, nodeTree) {
    summariesTree.find('ul.storedSummarybrowseTree > li').addClass('Expanded');
    summariesTree.find('ul.storedSummarybrowseTree > li > a').attr('aria-expanded','true');
    summariesTree.find('ul').find('li').find('ul').addClass('in').attr('aria-expanded','true').attr('style',''); 
    summariesTree.find('ul.storedSummarybrowseTree a[aria-expanded=true]').find('img').attr('src',expandIconUrl);
    addToExportPdfSummariesChapterExpandedNodes(initialExportPdfUrl, nodeTree);
    
    function addToExportPdfSummariesChapterExpandedNodes(initialExportPdfUrl, nodeTree) {
        //Convert stringified array to comma separated string.
        let expandedLinksCodes = nodeTree.replace(/(^\[|\]$|\s)/mg,'');        
        $('.PSPDF').attr('href', collectSummariesChapterNewExportPdfUrl(initialExportPdfUrl, expandedLinksCodes));
    };
}

function collapseAllSummariesChapterNodes(summariesTree, initialExportPdfUrl, collapseIconUrl) {
    summariesTree.find('li.Expanded > a').attr('aria-expanded','false');
    summariesTree.find('li.Expanded').removeClass('Expanded');
    summariesTree.find('ul').find('li').find('ul').removeClass('in').attr('aria-expanded','false').attr('style','height: 0px;');         
    summariesTree.find('a[aria-expanded=false]').find('img').attr('src',collapseIconUrl); 
    removeFromExportPdfSummariesChapterExpandedNodes(initialExportPdfUrl);
    
    function removeFromExportPdfSummariesChapterExpandedNodes(initialExportPdfUrl) {        
        $('.PSPDF').attr('href', initialExportPdfUrl);    
    };
}
//Get all expanded nodes
function collectSummariesChapterExpandedNodes(initialExportPdfUrl, summariesTree) {    
    let expandedLinks = summariesTree.children().find('a[aria-expanded=true]');        
    let expandedLinksCodes = [];      
    expandedLinks.each(function(idx, link) { expandedLinksCodes.push( $(link).prop('id').replace('arrow_','') ); });        
    $('.PSPDF').attr('href', collectSummariesChapterNewExportPdfUrl(initialExportPdfUrl, expandedLinksCodes.join(',') ));
}
//Rewrite the Export Pdf button url according to the Expanded links.
function collectSummariesChapterNewExportPdfUrl(initialExportPdfUrl, expandedLinksCodes) {
    let exportPdfUrl = decodeURIComponent(initialExportPdfUrl);
    if(expandedLinksCodes !== undefined && expandedLinksCodes.length != 0) {
        let splitExportPdfUrl = exportPdfUrl.split('&');
        return splitExportPdfUrl[0]+'&expand='+encodeURIComponent(expandedLinksCodes)+'&'+splitExportPdfUrl[1];      
    }
    else {
        return initialExportPdfUrl;
    }        
}


function toggleAllCaseFile(collapse) {
    const cdTimelineBlock = $(".cd-timeline__block");
    const instDetailsCollapse = cdTimelineBlock.find(".collapse");

    const embeddedAppeals = $(".embeddedAppealContent");
    const embeddedAppealsContainer = $(".embeddedAppealContainer");


    instDetailsCollapse.map(function(_, obj) {
        createCookie("CaseFile" + obj.id, (collapse ? '0' : '1'), 30);
    });

    instDetailsCollapse.collapse(collapse ? "hide" : "show");
    embeddedAppeals.collapse(collapse ? "hide" : "show");

    embeddedAppealsContainer.removeClass(collapse ? "open" : "")
                            .addClass(collapse ? "" : "open");

    cdTimelineBlock.find(".row > button > i.fa")
                       .removeClass(collapse ? "fa-minus-square" : "fa-plus-square")
                       .addClass(collapse ? "fa-plus-square" : "fa-minus-square");

    $(".embeddedAppealBtn > i.fa")
                   .removeClass(collapse ? "fa-minus-square" : "fa-plus-square")
                   .addClass(collapse ? "fa-plus-square" : "fa-minus-square");
}

function expandAllCaseFile()   { toggleAllCaseFile(false); }
function collapseAllCaseFile() { toggleAllCaseFile(true);  }

function dailyViewDatepicker(locale){

	initDatePicker(locale);

	// Used to validate date
	Date.prototype.isValid = function () {
		return this.getTime() === this.getTime();
	};

	// flag to know whether the datepicker accessed via tab
	var accessedViaTab = false;

	//flag goes true when the datepicker is opened with enter key
	$('#CalendardateExact ').keydown(function (e) {
		if (e.which === 13) { // Enter key
			accessedViaTab = true;
		}
	});
	//reset the flag if we accidentally access the datepicker with tab
	$('#CalendardateExact').find('.btn')
		.mousedown(function(ev){
			if(ev.which == 1)
			{
				ev.preventDefault();
				accessedViaTab = false;
			}
		});

	$('#CalendardateExact').datetimepicker()
		.on('dp.change', function(ev){

			ev.preventDefault();
			ev.stopPropagation();
			//the provided date may be invalid
			var selectedDate = new Date(ev.date);

			//submit the form when valid and not accesed via tab
			if(selectedDate.isValid() && accessedViaTab == false){
				$('#daily-view-oj-search-form').submit();
				//avoid double submission issue
				$('#Calendarbtn').find('button').prop('disabled', true);
			}
		});


	// when accessed via tab
	$('#dateExact').keydown(function (e) {
		if (e.which === 13) { // Enter key
			e.preventDefault();
			e.stopPropagation();
			//change format to YYYY/MM/DD in order to pass validation
			var dateInput= $('#dateExact').val();
			let dateArray = dateInput.split("/");
			let newDate = `${dateArray[2]}/${dateArray[1]}/${dateArray[0]}`;
			var selectedDate = new Date(newDate);
			if (selectedDate.isValid() ) {
				$('#daily-view-oj-search-form').submit();
				// Avoid double submission issue
				$('#Calendarbtn_dateExact button').prop('disabled', true);
			}
		}
	});

}
function toggleContent(label,labelId, labelTextId) {
	var content = label.nextSibling.nextSibling; // Get the next sibling, which is the content div
	var labelText = document.getElementById(labelId).textContent;
	var labelTextLang = document.getElementById(labelTextId).textContent.trim();
	var downArrow = '&#9207; '; // Downward-facing arrow
	var rightArrow = '&#9205; '; // Right-pointing arrow

	if (content.style.display === 'none' || content.style.display === '') {
		content.style.display = 'block';

		label.innerHTML =  downArrow + labelText ; // Change label removing languages

	} else {
		content.style.display = 'none';
		label.innerHTML = rightArrow +  labelText + labelTextLang; // Revert label text to the original state

	}
}
//EURLEX-5917 - Clear name and comment of alert
function clearFieldsForEmailAlert() {
    $("#itemName").val("");
    $("#itemComment").val("");
}
