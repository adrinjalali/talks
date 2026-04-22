// General PopOvers (start)
$('.EurlexPopover').popover({
	template: '<div class="popover" role="tooltip"><div class="arrow"></div><div class="popover-title"></div><div class="popover-content"></div></div>',
	html: true
})
// General PopOvers (end)

// General Tooltips (start)
$('.EurlexTooltip').tooltip()
// General Tooltips (end)

// Form Help Tooltips (start)
// Initialize tooltips.
$(function() {
    $('.FormHelpTooltip').tooltip({
        trigger: 'click hover',
        delay: { show: 100, hide: 200 },
        html: true
    });
});

// Configure tooltips.
$('.FormHelpTooltip').click(function(event) {
    event.preventDefault(); // Make tooltip anchor element's (<a></a>) link not clickable.
}).on('contextmenu', function() {
    const self = $(this);

    const tooltipData = self.data('bs.tooltip');
    if (tooltipData.inState && (tooltipData.inState.click === false)) {
        // Hide the tooltip on browser's context menu activation, to prevent (possible)
        // problems where the tooltips are not dismissed properly.
        self.tooltip('hide');
    }
});

// Keep only one tooltip visible each time.
$(document).on('click', function(event) {
	$('[data-toggle="tooltip"]').each(function() {
	    const self = $(this);
	    const tooltipShouldBeDismissed = !self.is(event.target) &&
	            (self.has(event.target).length === 0) &&
	            ($('.tooltip').has(event.target).length === 0);
		if (tooltipShouldBeDismissed) {
		    ((self.tooltip('hide').data('bs.tooltip') || {}).inState || {}).click = false;
		}
	});
});

// Keep tooltip visible on mouse-over.
const tooltipOriginalMouseLeaveEvent = $.fn.tooltip.Constructor.prototype.leave;
$.fn.tooltip.Constructor.prototype.leave = function(object) {
	const self = object instanceof this.constructor ?
		object :
		$(object.currentTarget)[this.type](this.getDelegateOptions()).data('bs.' + this.type);

    tooltipOriginalMouseLeaveEvent.call(this, object);

    if (object.currentTarget) {
		const container = $(object.currentTarget).siblings('.tooltip');
		container.one('mouseenter', function() {
			clearTimeout(self.timeout);
			container.one('mouseleave', function() {
				$.fn.tooltip.Constructor.prototype.leave.call(self, self);
			});
		}).one('contextmenu', function() {
		    // Hide the tooltip on browser's context menu activation, to prevent (possible)
            // problems where tooltips are not dismissed properly.
            // NOTE FOR IE11: The tooltip may not be dismissed before the browser's context
            // menu is activated, but after the context menu is also dismissed. However, this
            // behavior does not seem to cause further problems regarding tooltips.
		    $.fn.tooltip.Constructor.prototype.leave.call(self, self);
		});
	}
};
// Form Help Tooltips (end)

// Slide dropdowns (start)
$('.dropdown').on('show.bs.dropdown', function (e) {
	$(this).find('.dropdown-menu').first().stop(true, true).slideDown(300);
});
$('.dropdown').on('hide.bs.dropdown', function (e) {
	$(this).find('.dropdown-menu').first().stop(true, true).slideUp(200);
});
// Slide dropdowns (end)

// Disable QuickSearch and AdvancedSearch textarea 'enter' key (start)
$('#QuickSearchField, .AdvancedSearchTextarea, #FindByKeywordTextArea').on('keypress', function (e) {
	if ((e.keyCode || e.which) == 13) {
		$(this).parents('form').submit();
		return false;
	}
});
// Disable QuickSearch and AdvancedSearch textarea 'enter' key (end)

// Show/Hide QuickSearch options (start)
function ShowQSHelp() {
	$('div.QuickSearchOptions').fadeIn('medium').removeClass('sr-only').addClass('in');
	setQSHelpMargin();
}
function showQSHelpAlt() {
	$('div#QuickSearchHelp').html(quickSearchHelpAlt);
	ShowQSHelp();
}
function closeQSHelp() {
	$('div#QuickSearchHelp').html(quickSearchHelp);
	setQSHelpMargin();
	$('div.QuickSearchOptions').addClass('sr-only').removeClass('in');
}
function setQSHelpMargin() {
	var baseMargin = 5;
	var formHeight = $('#quick-search').outerHeight();
	var optionsHeight = $('.QuickSearchOptions').outerHeight();
	if (window.innerWidth < 768) {
		$('.QuickSearchOptions').css('margin-top', baseMargin);
	} else {
		$('.QuickSearchOptions').css('margin-top', -(baseMargin+formHeight+optionsHeight));
	}
}
$('#QuickSearchField').on('focus', ShowQSHelp);
$('.QuickSearchBtn').on('keydown', closeQSHelp);
$(window).resize(function() {
	setQSHelpMargin();
	if(typeof initConslegTimelineCustom === 'function') {
		initConslegTimelineCustom();
	}
});
// Show/Hide QuickSearch options (end)

// Distinctive Quick Searches (start)
$(document).ready(function () {
	initRelatedForms();
});

function initRelatedForms() {
	var $forms = $('.DistinctiveForm'),
		isActivated = false;

	var isHomePage = window.location.pathname.indexOf("homepage") >= 0;

	if (isHomePage) {
		$("input, textarea", $forms).each(function () {
			$currentForm = $(this).closest('.DistinctiveForm');

			if (!($currentForm.attr('id') === ('quick-search') || this.type === 'hidden')) {
				if (this.value) {
					disableFormsFields($currentForm);
				}
			}
		});
	} else {
		$("input, textarea", $forms).each(function () {
			$currentForm = $(this).closest('.DistinctiveForm');

			if (!($currentForm.attr('id') === ('quick-search') || this.type === 'hidden')) {
				if (this.value) {
					var QSform = $('.DistinctiveForm.QSF').get();
					disableFormFields($currentForm, QSform);
					$('div.QuickSearchOptions').fadeIn('medium').removeClass('sr-only').addClass('in');
					$(document).on('focusin.QuickSearchOptions click.QuickSearchOptions', function (e) {
						if ($(e.target).closest('.QuickSearchOptions, #QuickSearchField, .QuickSearchBtn').length) return;
						$(document).unbind('.QuickSearchOptions');
						$('div.QuickSearchOptions').addClass('sr-only').removeClass('in');
					});
				}
			}
		});
	}

	if ($forms.length < 1)
		return;

	$("input, textarea", $forms).on('input', function () {
		var $this = $(this),
			value = this.value,
			$currentForm = $this.closest('.DistinctiveForm'),
			isQuickSearch = ($currentForm.hasClass('QSF')) ? true : false;


		if (value === "") {
			if (isEmptyFields($currentForm)) {
				enableFormsFields($currentForm);
				if (!isQuickSearch) {
					isActivated = false;
				}
			}
		} else {
			disableFormsFields($currentForm);
			if (!isQuickSearch) {
				isActivated = true;
			}
		}

	});

	$("select", $forms).change(function () {
		var $this = $(this),
			index = this.selectedIndex,
			$currentForm = $this.closest('.DistinctiveForm'),
			isQuickSearch = ($currentForm.hasClass('QSF')) ? true : false;

		if (index == 0) {
			if (isEmptyFields($currentForm)) {
				enableFormsFields($currentForm);
				if (!isQuickSearch) {
					isActivated = false;
				}
			}
		} else {
			disableFormsFields($currentForm);
			if (!isQuickSearch) {
				isActivated = true;
			}
		}

	});

	$('html').click(function (event) {
		if (!$(event.target).closest('.QuickSearchOptions').length && !$(event.target).closest('.QSF').length && !$(event.target).closest('.FindResultsBy').length) {
			if (isActivated)
				clearQSFForms();
		}
	});

	function isEmptyFields($form) {
		var isClear = true;

		$("input:visible, textarea", $form).each(function () {
			var $this = $(this);

			if ($this.val() !== '') {
				isClear = false;
				return false;
			}
		});

		$("select", $form).each(function () {
			if (this.selectedIndex != 0) {
				isClear = false;
				return false;
			}

		});

		return isClear;
	}

	function disableFormFields($exceptForm1, $exceptForm2) {
		$('.DistinctiveForm').not($exceptForm1).not($exceptForm2).find('input, textarea, select, button').prop('disabled', true);
		$('.DistinctiveForm').not($exceptForm1).not($exceptForm2).find('a').css('visibility', 'hidden');
		$('.DistinctiveForm').not($exceptForm1).not($exceptForm2).find('.DistinctiveFormMessage').fadeIn();
	}

	function disableFormsFields($exceptForm) {
		$('.DistinctiveForm').not($exceptForm).find('input, textarea, select, button').prop('disabled', true);
		$('.DistinctiveForm').not($exceptForm).find('a').css('visibility', 'hidden');
		$('.DistinctiveForm').not($exceptForm).find('.DistinctiveFormMessage').fadeIn();
	}

	function enableFormsFields($exceptForm) {
		$('.DistinctiveForm').not($exceptForm).find('input, textarea, select, button').prop('disabled', false);
		$('.DistinctiveForm').not($exceptForm).find('a').css('visibility', '');
		$('.DistinctiveForm').not($exceptForm).find('.DistinctiveFormMessage').fadeOut();
	}

	/* Applies to 'Quick Search' form and 'Find results by' widgets.
	 * If any point in the screen is clicked, enable them if disabled, and clear their content (except for hidden input elements).
	 * Exception: If the 'Quick Search' textarea has a value, it isn't cleared and the 'Find results by' widgets remain disabled.
	 */
	function clearQSFForms() {
		if ($('.QSF').find('textarea').val() == ''){
			$('.DistinctiveForm').each(function () {
				var $form = $(this);
				$("select", $form).prop('disabled', false).find('option').eq(0).prop('selected', true);
				$("input, textarea, button", $form).prop('disabled', false);
				$("input[type!='hidden']", $form).val('');
				$('.DistinctiveForm').find('a').css('visibility', '');
				$('.DistinctiveForm').find('.DistinctiveFormMessage').fadeOut();
			});
		}
	}
}

// Distinctive Quick Searches (end)

// Enable links in TreeMenu with <span class="TMLink"> (start)
$(".TMLink").click(function (e) {
	e.stopPropagation();
	window.location = $(this).parent().attr('href');
});
// Enable links in TreeMenu with <span class="TMLink"> (end)

// Initialize tree menu and News page left menu (start)
if ($(".TreeMenu").length) {
	$(".TreeMenu").metisMenu({
		activeClass: 'Expanded'
	});
}
if ($(".CompactStaticMenu").length) {
	$(".CompactStaticMenu").metisMenu({
		activeClass: 'Expanded'
	});
}

$(window).on('resize load', function () {
	if ($(".TreeMenu").length) {
			$(".NoAccordionTreeMenu").metisMenu({
				activeClass: 'Expanded',
				toggle: false
			});
	}
	if ($(".CompactStaticMenu").length) {
		$(".NoAccordionTreeMenu").metisMenu({
			activeClass: 'Expanded',
			toggle: false
		});
}
});
// Initialize tree menu and News page left menu (end)

// Expert Search Trees (start)
$(".ExpertSearchTree").metisMenu({
	toggle: false,
	activeClass: 'Expanded'
});

$("#ExpandAllTree").click(function (e) {
	$('.ExpertSearchTree .has-arrow').parent('li').addClass('Expanded');
	$('.ExpertSearchTree a[aria-expanded="false"]').attr('aria-expanded', 'true');
	$('.ExpertSearchTree ul[aria-expanded="false"]').attr('aria-expanded', 'true').addClass('in').removeAttr('style');
});
$("#CollapseAllTree").click(function (e) {
	$('.ExpertSearchTree .has-arrow').parent('li').removeClass('Expanded');
	$('.ExpertSearchTree a[aria-expanded="true"]').attr('aria-expanded', 'false');
	$('.ExpertSearchTree ul[aria-expanded="true"]').attr('aria-expanded', 'false').removeClass('in');
});

$(".ExpertSearchValueTree").metisMenu({
	toggle: false,
	activeClass: 'Expanded'
});

// Expert Search Trees (end)


// Set various homepage equal heights (start)
$(window).on('resize load', function () {
	var windowSize = window.innerWidth;

	if (windowSize > 991) {
		$(".HomeOJ").css({'min-height': ($(".NavSearchHome").innerHeight() + $(".Promo").innerHeight() + 50 + 'px')});
		$(".MenuBlock2").css({'min-height': ($(".MenuBlock3").innerHeight() + 1 + 'px')});
		$(".MenuBlock1").css({'min-height': ($(".MenuBlock3").innerHeight() + $(".MenuBlock4").innerHeight() + 24 + 'px')});
		$("#QSByDocNumber .panel-body").css({'min-height': ($(".MenuBlock3").innerHeight() - $("#QSByCelexTitle").innerHeight() - 30 + 'px')});
		$("#QSByCelex .panel-body").css({'min-height': ($(".MenuBlock3").innerHeight() - $("#QSByDocNumberTitle").innerHeight() - 30 + 'px')});
	} else if (windowSize > 767) {
		$(".HomeOJ").css({'min-height': ($(".FindResultsBy").innerHeight() + 'px')});
		$(".MenuBlock1").css({'min-height': ($(".MenuBlock2").innerHeight() + $(".MenuBlock3").innerHeight() + 25 + 'px')});
		$("#QSByCelex .panel-body").css({'min-height': ($(".HomeOJ").innerHeight() - $("#QSByDocNumberTitle").innerHeight() - 33 + 'px')});
	} else {
		$(".HomeOJ").css({'min-height': 'auto'});
		$(".MenuBlock2").css({'min-height': 'auto'});
		$(".MenuBlock1").css({'min-height': 'auto'});
		$("#QSByCelex .panel-body").css({'min-height': 'auto'});
	}
});
// Set various homepage equal heights (end)

// Collapse panel on-resize in search results and display ViewMoreInfo button correctly (start)
$(window).on('resize', function () {
    if (window.innerWidth < 992) {
        $('.SearchResult .CollapsePanel-sm').each(function (i, e) {
        	var aria = $('.ViewMoreInfo', e).attr("aria-expanded");
            if (aria == 'true') {
                $('.collapse', e).addClass('in');
            } else {
                $('.collapse', e).removeClass('in');
            }
        });
    } else {
        $('.SearchResult .CollapsePanel-sm .collapse').addClass('in');
        $('.SearchResult .CollapsePanel-sm .panel-title a').removeClass('collapsed');
        $('.SearchResult .CollapsePanel-sm .panel-title a').attr('aria-expanded', 'true');
    }
});
// Collapse panel on-resize in search results and display ViewMoreInfo button correctly (end)

// Megamenu (start)
// Set Megamenu width
$(window).resize(function () {
	$(".MegaMenu").css({'width': ($(".NavSearch").width() + 'px')});
});
$(window).trigger('resize');

// Keep Megamenu open when clicking on it
$('.MegaMenu').on('click', function (e) {
	e.stopPropagation();
});

// Close Megamenu when focus is on the Quick Search (for keyboard users)
$('#QuickSearchField').focusin(function () {
	$('.MegaMenu').slideUp(200);
});
// Megamenu (end)

// Offcanvas menu (start)
$(document).ready(function () {
	$('[data-toggle="offcanvas"]').click(function () {
		$('.row-offcanvas').toggleClass('active')
	});
    $('#helpMenu1 a.faqLink,#helpMenu1 span.TMLink,#helpMenu2 span.TMLink').click(function () {
    	//Deactivate transition
        $(".row-offcanvas").addClass('notransition');
        //remove active class
        $('.row-offcanvas').toggleClass('active');
        //activate transition
        $(".row-offcanvas")[0].offsetHeight; // Trigger a reflow, flushing the CSS changes
        $(".row-offcanvas").removeClass('notransition');
    });
});
// Offcanvas menu (end)

// Expand collapse all page panels (start)
$("#ExpandAll").click(function (e) {
	$(".PagePanel .panel-collapse").not(".childPanel").collapse("show");
	$(".AdvancedSearchPanel .panel-collapse").not(".childPanel").collapse("show");
	$(".panelOjAba .panel-collapse").not(".childPanel").collapse("show");
});

$("#CollapseAll").click(function (e) {
	$(".PagePanel .panel-collapse").not(".childPanel").collapse("hide");
	$(".AdvancedSearchPanel .panel-collapse").not(".childPanel").collapse("hide");
	$(".panelOjAba .panel-collapse").not(".childPanel").collapse("hide");
});
// Expand collapse all page panels (end)

// Diable click on unavailable publication format languages (start)
$(".PubFormat .disabled a").click(function (event) {
	event.preventDefault();
});
// Diable click on unavailable publication format languages (end)

// Affix sidebar (start)
/* Separate call for document.ready to avoid affix sidebar stuck on bottom, when page is scrolled to bottom and refreshed
 * Doesn't work in Firefox, extra click needed.
 */
$(document).ready(function () {
	setAffixSidebar();
	$(document).click();
});
$(window).on('resize', function () {
    if (!(/TXT\/HTML/.test(window.location.href) && window.innerWidth <= 991)){setAffixSidebar();}
});

function setAffixSidebar() {
	var globanHeight = 0;
	var consentHeight = 0;
	if($("div.container-fluid").length && cookiesNoChoice()) {
		consentHeight = 109;
	}
	if($('.globan') != null && $('.globan').height() != null) {
		globanHeight = $('.globan').height();
	}
	var extraCalculatedHeight = globanHeight + consentHeight;
	var headerHeight = $('header').outerHeight() + $('.NavSearch').outerHeight() + $('.SiteBreadcrumb').outerHeight() + $('.PageTitle').outerHeight() + (extraCalculatedHeight)-20;
	var footerHeight = $('footer').outerHeight() + 40;


	if (window.innerWidth > 991) {
		if($('#AffixSidebar').affix().data('bs.affix')) {
			$('#AffixSidebar').affix().data('bs.affix').options.offset = {
				top: headerHeight,
				bottom: footerHeight
			}		
			// Update SidebarWrapper width (in desktop)
			$('#AffixSidebar').width($('.AffixSidebarWrapper').width() - 10);
		}
	}
	// Update SidebarWrapper width (in mobile)
	$('#AffixSidebar').width($('.AffixSidebarWrapper').width());	
}
//Affix sidebar (end)

/* MODALS CODE  */

// Modal body scroll (start)
function setModalMaxHeight(element) {
	// alert("setModalMaxHeight() called")
	this.$element = $(element);
	this.$content = this.$element.find('.modal-content');
	var borderWidth = this.$content.outerHeight() - this.$content.innerHeight();
	var dialogMargin = window.innerWidth < 768 ? 20 : 60;
	var contentHeight = $(window).height() - (dialogMargin + borderWidth);
	var headerHeight = this.$element.find('.modal-header').outerHeight() || 0;
	var footerHeight = this.$element.find('.modal-footer').outerHeight() || 0;
	var fixedContentHeight = this.$element.find('.FixedModalContent').outerHeight() || 0;
	var modalActionsHeight = this.$element.find('.ModalActions').outerHeight() || 0;
	var maxHeight = contentHeight - (headerHeight + footerHeight + fixedContentHeight + modalActionsHeight);

	this.$content.css({
		'overflow': 'hidden'
	});

	this.$element
		.find('.modal-body').css({
		'max-height': maxHeight,
		'overflow-y': 'auto',
		'overflow-x': 'hidden'
	});

	this.$element
		.find('.modal-content').css({
		'padding-bottom': modalActionsHeight
	});
}

// This event fires immediately when the show instance method is called.
// If caused by a click, the clicked element is available as the relatedTarget property of the event.
$('.modal').on('show.bs.modal', function () {
	// alert("show.bs.modal triggered")
	$(this).show();
	setModalMaxHeight(this);
});

/*Hide modal - related functionalities */
// This event is fired when the modal has finished being hidden from the user (will wait for CSS transitions to complete).
$('#myModal').on('hidden.bs.modal', function (e) {
	modalCleanup();
})

$(window).resize(function () {
	if ($('.modal.in').length != 0) {
		setModalMaxHeight($('.modal.in'));
	}
	/* ONLY for the Webservice Template modal: Override bootstrap default that sets width:auto for xs-screen,
	 * as it causes the modal to drop below the viewport, because of overflow in the <pre> child element.
	 * Margin value=168 selected to work smoothly with the fixed width:600px for larger screens.*/
	if (window.innerWidth < 768) {
		$(".modal-dialog:has(.singleInline64)").width(window.innerWidth - 168);
	}
	else{
		$(".modal-dialog:has(.singleInline64)").width(600);
	}
});
// Modal body scroll (end)


// Set defaults of spinner plugging
$(document).ready(function () {
	if ($.LoadingOverlaySetup) {
		$.LoadingOverlaySetup({
			color: "rgba(17, 34, 68, 0.8)",
			image: "",
			fontawesome: "fa fa-spinner fa-spin spinner-font ",
			fade: [400, 10]
		});
	}
});

/* END MODALS CODE  */


// Back to top (start)
$(document).ready(function () {
	var offset = 300;
	var duration = 300;
	$(window).scroll(function () {
		if ($(this).scrollTop() > offset) {
			$('.EurlexTop').fadeIn(duration);
		} else {
			$('.EurlexTop').fadeOut(duration);
		}
	});
	$('.EurlexTop').click(function (event) {
		event.preventDefault();
		$('html, body').animate({scrollTop: 0}, duration);
		return false;
	})
});

$(window).on('resize load', function () {
	$(".EurlexTop").css({'right': (($(window).width() - $("footer").innerWidth()) / 2 + 'px')});
});
// Back to top (end)

// Custom input type="file" (start)
$(function () {
	$(document).on('change', ':file', function () {
		var input = $(this),
			numFiles = input.get(0).files ? input.get(0).files.length : 1,
			label = input.val().replace(/\\/g, '/').replace(/.*\//, '');
		input.trigger('fileselect', [numFiles, label]);
	});

	$(document).ready(function () {
		$(':file').on('fileselect', function (event, numFiles, label) {

			var input = $(this).parents('.input-group').find(':text'),
				log = numFiles > 1 ? numFiles + ' files selected' : label;

			if (input.length) {
				input.val(log);
			} else {
				if (log) alert(log);
			}
		});
	});
});
// Custom input type="file" (end)

// AutoGrow Textareas (start) [see plugin]
$(document).ready(
	function () {
		if ($('.AutoGrow').length) {
			$('.AutoGrow').autoResize({minRows: 1, maxRows: 5});
		}
	});
// AutoGrow Textareas (end)

// Add gradient background to responsive tables when their content becomes scrollable
$(window).on('resize load', function () {
	$('.table-responsive').each(function(){
		if($(this)[0].scrollWidth > $(this)[0].clientWidth){
			updateGradients(this, "rightGradient");
	    }
	    else{
	    	updateGradients(this, null);
	    }
	});
});
$('.table-responsive').on('scroll', function() {
	//If table content is scrollable
	if($(this).innerWidth() < $(this)[0].scrollWidth) {
		if($(this).scrollLeft() == 0){
	    	updateGradients(this, "rightGradient");
	    }	
		else if($(this).scrollLeft() + $(this).innerWidth() == $(this)[0].scrollWidth){
	    	updateGradients(this, "leftGradient");
	    }
		else{
			updateGradients(this, "bothGradients");
		}
	}
});
//For the 'View/Hide Search button' in My Saved Searches, which alters the table's scrollWidth
$('.nowrap.fieldHelp').click(function() {
	var table = $(this).closest('.table-responsive');
	if($(table).innerWidth() < $(table)[0].scrollWidth) {
		if ($(table).scrollLeft() == 0){
			updateGradients(table, "rightGradient");
		}
	}
	else {
		updateGradients(table, null);
	}
});

function updateGradients(table, gradientToAdd){
	$(table).removeClass("bothGradients");
	$(table).removeClass("rightGradient");
	$(table).removeClass("leftGradient");
	if (gradientToAdd != null){
		$(table).addClass(gradientToAdd);
	}
}
// Add gradient background to responsive tables when their content becomes scrollable(end)

// EURLEXNEW-3970 - ECB - Adjust height of first row widgets so as on expand/collapse of the tree to have borders on the same level. (start)
$(document).ready(function () {
	if ($("#EcbMenuBlock1").length)	{
		// Initialize min height of tree.
		var treeDivMinHeight = $("#EcbMenuBlock1").outerHeight();
		$("#EcbMenuBlock2").css(
				{'min-height': (treeDivMinHeight + 'px')});
		// Callback to be executed on resizing of tree.
		var treeResizeCallback = function() {
			updateStatisticsWidget($("#EcbMenuBlock1").outerHeight());
		};
		
		// "treeResize" events are automatically triggered upon
		// completion of transition effects of tree node expanding/collapsing
		// from functions 'smartToogleTree(el)' 'andloadNextLevel(el, ...)'.
		$("#EcbMenuBlock1").on("treeResize", treeResizeCallback);
		
		// Event to dynamically enable/disable resizing of statistics widget.
		$(window).on("load resize", function() {
			var treeDivWidth = 	$("#EcbMenuBlock1").parent().outerWidth();
			// Tree widget extends to the full width of the row ('xs' view).
			if ($(window).width() == treeDivWidth) {
				$("#EcbMenuBlock1").off("treeResize");
				updateStatisticsWidget(treeDivMinHeight);
			}
			else {
				$("#EcbMenuBlock1").on("treeResize", treeResizeCallback);
				$("#EcbMenuBlock1").trigger("treeResize");
			}
		});
	}
});

function updateStatisticsWidget(treeHeight) {
	$("#EcbMenuBlock2").css(
		{'min-height': (treeHeight + 'px')});
	
	// Expand carousel if 2 graphs can fit in the containing block.
	var blockHeight = $('#EcbMenuBlock1').height()-40;
	var titleHeight = $('#EcbMenuBlock2 h2.Ecb-blockTitle').outerHeight(true);
	var graphHeight = $('#EcbMenuBlock2 .item.active .col-xs-12').height();
	var canExpand = (blockHeight - titleHeight > (2 * graphHeight));
	
	if ($('#statisticsCarousel').hasClass('collapsed') && canExpand === true){
		$('#statisticsCarousel').addClass('expanded');
		$('#statisticsCarousel').removeClass('collapsed');
		$('.cloneditem').each(function(){
			$(this).removeClass('hidden');
		});
	} else if ($('#statisticsCarousel').hasClass('expanded') && canExpand === false){
		$('#statisticsCarousel').addClass('collapsed');
		$('#statisticsCarousel').removeClass('expanded');
		$('.cloneditem').each(function(){
			$(this).addClass('hidden');
		});
	}
	
	// Adjust graph top and bottom padding so that they always stay vertically centered.
	var paddingEach;
	if ($('#statisticsCarousel').hasClass('collapsed')){
		var paddingTotal = blockHeight - titleHeight - graphHeight;
		paddingEach = Math.floor(paddingTotal / 2);
	} else if ($('#statisticsCarousel').hasClass('expanded')){
		var paddingTotal = blockHeight - titleHeight - (2 * graphHeight);
		paddingEach = Math.floor(paddingTotal / 4);
	}
	$('#statisticsCarousel .col-xs-12').each(function(){
		$(this).css('padding-top', paddingEach+'px');
		$(this).css('padding-bottom', paddingEach+'px');
	});
}

//EURLEXNEW-3979 - Method used for accessing help video
function accessHelpVideo(event, path) {
       event.preventDefault();
       window.location.href = path + '#video';
}

//EURLEXNEW-3970 - ECB - Adjust height of first row widgets... (end)
//EURLEXNEW-3970 - ECB - Statistics widget onclick event.
$(document).ready(function () {
	$('#statisticsCarousel .carousel-inner').click(function(e) {
		window.location.href = 'bank/statistics.html';
	});
});


function enableDateRangeSection(){
	$("#ojYears").attr('disabled', true);
  	$("#ojSeries").attr('disabled', true);
  	$("#ojNumber").attr('disabled', true);
    
  	$("#dateFrom").attr('disabled', false);
  	$("#dateTo").attr('disabled', false);
}

function enableOjReferenceSection(){
	$("#ojYears").attr('disabled', false);
  	$("#ojSeries").attr('disabled', false);
  	$("#ojNumber").attr('disabled', false);
    
  	$("#dateFrom").attr('disabled', true);
  	$("#dateTo").attr('disabled', true);
}

//EURLEXNEW-4201
//This method is also called on resize to re-calculate the timeline
function initConslegTimelineCustom() {
	  if(!$('#custom_timeline').text()){return;}
	  $('#custom_timeline').find('.line').remove();
	  	  
	  var basicActLink = $('#custom_timeline #basicActLink a').attr('href');
	  var activeVersion = $('#custom_timeline #dateSelected').text();
	  var futureDate = $('#custom_timeline  #futureDate').text();
	  var currentDate = $('#custom_timeline #dateCurrent').text();
	  var afterSelectedDate = $('#custom_timeline #dateSelected').prev().text();
	  var showBreakLineBeforeCurrentAct = afterSelectedDate != currentDate;	  
	  var selectedItem =  $('#custom_timeline #dateSelected');
	  var selectedIsCurrent = selectedItem.text().length < 3;
	  
	  if(selectedIsCurrent){
		  selectedItem = $('#custom_timeline #dateCurrent');
	  }
	  
	  var beforeSelectedDate = selectedItem.next().text();
	  var afterBasicActDate = $('#custom_timeline #basicAct').prev().text();
	  
	  var showBreakLineAfterBasicAct = beforeSelectedDate != afterBasicActDate;
	  if(selectedItem.text() == afterBasicActDate){
		  showBreakLineAfterBasicAct = false;
	  }
	  
	  var selectedAndCurrentSame = currentDate == activeVersion;
	  
	  var showPrevBreakline = false;
	  var showNextBreakLline = false;
	  var isPreviousOfSelectedBasicAct = selectedItem.text() == afterBasicActDate;
	  var isNextOfSelectedCurrentVersion = afterSelectedDate == currentDate;
	  var isActiveAndSelectedSameDate = false;
	  var isFutureDateAndActiveDateSame = futureDate == activeVersion;
	  var noFutureDate = $("#custom_timeline .futureDate").text().length == 0;
	  
	  //Define the available line width.
	  var availableWidth = $("#custom_timeline").width()-50;
	  
	  //business logic here to adapt list, by removing stuff
	  var distanceFromBasicToNextNoHiddenWidth =  (10 * availableWidth / 100);
	  
	  //if previous of selectedVersion is the first one/basic act
	  //then the line is NO Breakline and the line width is from the first/basicAct to the next one that is not hidden.
	  if(isPreviousOfSelectedBasicAct) {
		  distanceFromBasicToNextNoHiddenWidth = $("#custom_timeline .basicAct").prevAll(':not(.hidden)').filter(':first').offset().left - $(".basicAct").offset().left;
	  } 
	  
	  $("#custom_timeline .basicAct").append("<div class='line' style='width: " + distanceFromBasicToNextNoHiddenWidth +"px'><span class='break-arrow'></span></div>");
	  if(showBreakLineAfterBasicAct) {
		  $("#custom_timeline .basicAct .line").addClass('breakline');
	  }
	  	  
	  //Compute diffs,lineWidth and percentages
	  var diff1 = parseInt(selectedItem.prev().attr("data-timestamp")) - parseInt(selectedItem.next().attr("data-timestamp"));
	  var diff2 = parseInt(selectedItem.prev().attr("data-timestamp")) - parseInt(selectedItem.attr("data-timestamp"));
	  var diff3 = parseInt(selectedItem.attr("data-timestamp")) - parseInt(selectedItem.next().attr("data-timestamp"));	  
	  
	  var lineWidth = selectedItem.offset().left - selectedItem.nextAll(':not(.hidden)').filter(':first').offset().left;
	  
	  var perc1 = diff3 * 100 / diff1;
	  var perc2 = 10 + 50*perc1/100;
	  
	  // Showing and placing PREVIOUS of selected act
	  selectedItem.next().removeClass('hidden');
	  selectedItem.next().attr("style", "left: " + (10 * availableWidth / 100) + "px");
	  
	  // Calculate width line IF previous not basic act
	  if(!isPreviousOfSelectedBasicAct) {
		  selectedItem.next().append("<div class='line' style='width: " + lineWidth +"px'></div>");
	  }
	  
	  // Showing and placing NEXT of selected act
	  selectedItem.prev().removeClass('hidden');
	  selectedItem.prev().attr("style", "left: " + (60 * availableWidth / 100) + "px");

	  // Placing selected item
	  selectedItem.attr("style", "left: " + (perc2 * availableWidth / 100) + "px");
	  var lineWidth = (perc2 * availableWidth / 100) - (10 * availableWidth / 100);

	//calculate safe distance selected from previous
	var distanceSelectedFromPrev =   (perc2 * availableWidth / 100) - (10 * availableWidth / 100);
	if(distanceSelectedFromPrev<70){
		selectedItem.attr("style", "left: " + (perc2 * availableWidth / 100 + 60)   + "px");
	}
	// calculate safe distance selected from next
	var distanceSelectedFromNext = (60 * availableWidth / 100) - (perc2 * availableWidth / 100);
	if(distanceSelectedFromNext<70){
		selectedItem.attr("style", "left: " + (perc2 * availableWidth / 100 -80)   + "px");
	}

	  if(!selectedIsCurrent){
		  selectedItem.prev().append("<div class='line' style='width: " + lineWidth +"px'></div>");
	  }
	  	  
	  
	  var selLineWidth =  (60 * availableWidth / 100) - (perc2 * availableWidth / 100) + 10;
	  if(isNextOfSelectedCurrentVersion) {
		  selLineWidth = $("#custom_timeline .dateCurrent").offset().left - selectedItem.offset().left;
	  } 
	  
	  if(!selectedIsCurrent){
		  selLineWidth = selectedItem.prevAll(':not(.hidden)').filter(':first').offset().left -
	  	  selectedItem.offset().left;
		  selectedItem.append("<div class='line selLine' style='width: " + selLineWidth +"px'></div>");
	  }

	  if(showNextBreakLline && !selectedIsCurrent) {
		  selectedItem.next().find('.line').addClass('breakline');
	  }
	  
	  var selectedTwotBeforeCurrent = selectedItem.prev().prev().text() == currentDate;
	  
	  if(showBreakLineBeforeCurrentAct && !selectedTwotBeforeCurrent) {
		  
		  addBreakLineBeforeCurrentAct(selectedItem);
		  
	  }	 
	  
	  $("#custom_timeline .futureDate").attr("style", "left: " + availableWidth + "px");	  

	  var futureCurrentDifference = parseInt($("#custom_timeline .futureDate").attr("data-timestamp")) - parseInt($("#custom_timeline .dateCurrent").attr("data-timestamp"));
	  var todayCurrentDifference = parseInt($("#custom_timeline .today").attr("data-timestamp")) - parseInt($("#custom_timeline .dateCurrent").attr("data-timestamp"));
	  
	  var perc1 = todayCurrentDifference * 100 / futureCurrentDifference;
	  var perc2 = 100 - perc1;
	  
	  
	  if(noFutureDate){
		  perc2 = recalculatePercentageWhenNoFutureDate(availableWidth,selectedItem,selectedIsCurrent);
	  }
	  
	  $("#custom_timeline .today").attr("style", "left: " + (perc2 * availableWidth / 100) + "px");
	  var lineWidth = (perc2 * availableWidth / 100) - (50 * availableWidth / 100);  
	  
	  if(isNextOfSelectedCurrentVersion && noFutureDate){
			var lineWidth =  $("#custom_timeline .today").offset().left - $("#custom_timeline .dateCurrent").offset().left 
			$("#custom_timeline .dateCurrent").find('.line').attr("style", "width: " + lineWidth + "px")
	  }
	  
	  if(!isNextOfSelectedCurrentVersion){
		  $("#custom_timeline .dateCurrent").attr("style", "left: " + (70 * availableWidth / 100) + "px");
		  var lineCurrentToToday =  $("#custom_timeline .today").offset().left - $("#custom_timeline .dateCurrent").offset().left 
		  $("#custom_timeline .dateCurrent").append("<div class='line' style='width: " + lineCurrentToToday +"px'></div>");
		  
	  }
	 
	  var selLineWidth =  (100 * availableWidth / 100) - (perc2 * availableWidth / 100);
	  
	  $("#custom_timeline .today").append("<div class='line' style='width: " + selLineWidth +"px'></div>");

	  if(selectedIsCurrent){	  
		  recalculatePercentagesAndWidthsWhenSelectedIsCurrent(selectedItem);  
	  }
	  
	  computeAndUseOfSafeDistance()
	  
}

//Compute a safe distance.
//This is used when one act is very close to the basic act.
function computeAndUseOfSafeDistance(){
	var safeDistance = $("#custom_timeline .basicAct").prevAll(':not(.hidden)').filter(':first').offset().left
	  - $("#custom_timeline .basicAct").offset().left ;
	  	  
	if(safeDistance < 10 ){
		  $("#custom_timeline .basicAct").prevAll(':not(.hidden)').filter(':first').attr("style", "left: " + 120 + "px");
	}
}

//Recalculate percentages and linewidths when the selected act is the current one.
function recalculatePercentagesAndWidthsWhenSelectedIsCurrent(selectedItem){
	var todayCurrentDifference = parseInt($("#custom_timeline .today").attr("data-timestamp")) - parseInt($("#custom_timeline .dateCurrent").attr("data-timestamp"));
	var currentBeforeDifference = parseInt($("#custom_timeline .today").attr("data-timestamp")) - parseInt($("#custom_timeline .dateCurrent").next().attr("data-timestamp"));
	  
	var perc1 = todayCurrentDifference * 100 / currentBeforeDifference;
	var perc2 = 100 - perc1;		  
	var width = $("#custom_timeline .today").position().left;

	if (todayCurrentDifference < 0){
		$("#custom_timeline .dateCurrent").attr("style", "left: " + (perc2 * width / 100 -50) + "px");
	}
	if(currentBeforeDifference <50){
		$("#custom_timeline .dateCurrent").attr("style", "left: " + (perc2 * width / 100 +50) + "px");

	}

	var lineWidth =  $("#custom_timeline .today").offset().left - $("#custom_timeline .dateCurrent").offset().left
	  
	selectedItem.find('.line').attr("style", "width: " + lineWidth + "px")
}

//Recalculate the percentage when there is no future date.
function recalculatePercentageWhenNoFutureDate(availableWidth,selectedItem,selectedIsCurrent){
	var tStampDiff1 = availableWidth - $("#custom_timeline .dateCurrent").position().left;
	var tStampDiff2 = $("#custom_timeline .today").position().left - selectedItem.position().left;;
	if(!selectedIsCurrent){
		  tStampDiff2 = selectedItem.prev().position().left - selectedItem.position().left;
	}
	var  perc1 = tStampDiff2 * 10 / tStampDiff1;
	return  perc2 = 100 - perc1;
}


//Case where we need to insert breakline before Current Act.
function addBreakLineBeforeCurrentAct(selectedItem){
	var breakLineWidth = $("#custom_timeline .dateCurrent").offset().left - 
	$("#custom_timeline .dateCurrent").nextAll(':not(.hidden)').filter(':first').offset().left;

selectedItem.prev().find('.line').addClass('breakline');
selectedItem.prev().find('.line').attr("style", "width: " + breakLineWidth + "px")
}

$(document).ready(function(){
  $("[id *= 'showAll']").click(function(){
    var siblings = $(this).siblings();
    siblings.show();
    $(this).hide();
  });
});

$(document).ready(function(){
  $("[id *= 'showAll']").click(function(){
    var siblings = $(this).siblings();
    siblings.show();
    $(this).hide();
  });
});

$(document).ready(function(){
  $("[id *= 'showAll']").click(function(){
    var siblings = $(this).siblings();
    siblings.show();
    $(this).hide();
  });
});

/* OPEC-936: 'Expand'/'Collapse' is not Synchronizing */
/**
 *  Toggles the icon ('plus'/'minus' icon) of a VMI button.
 *
 *  @param  buttonElement  The plain HTML element representing the
 *                         button. Generally, this should be equal
 *                         to 'this' when using the 'onclick' HTML
 *                         event bind attribute.
 */
function toggleVMIButtonIcon(buttonElement) {
    const button = $(buttonElement);

    // Retrieve the element whose 'collapse' behavior is controlled
    // by this button.
    const controlledElement = $("#" + button.attr("aria-controls"));

    // Only change the button's icon when the 'collapse' event for
    // the controlled element has been completed.
    if (!controlledElement.hasClass("collapsing")) {
        button.find("i.fa")
              .toggleClass(["fa-plus-square", "fa-minus-square"]);
    }
}

$(function(){
    if(document.querySelectorAll('input[name="selectedDocument"]:checked').length == $(".SearchResult").length){
        $( "input[id*='SelectAllResults']" ).prop('checked',true);
}});

$(window).on('load', function() {
    $("#globan").wrap("<div class='globan-background'></div>");
});

$('#expFeatHelp').tooltip({html:true});
