const eliSubdivisions = eliSubdivisionLabels ? Object.keys(eliSubdivisionLabels) : [];

//Filters by type of document
const filtersByTypeOfDocument = new Map([
  ['CONSLEG', []],
  ['OJ', []]
]);

//CSS Styles by type of document
const stylesByTypeOfDocument = new Map([
  ['CONSLEG', new Map([
    ['enc', ['bold']],
    ['art', ['bold']],
    ['cpt', ['italic']],
    ['prt', ['italic']],
    ['tis', ['italic']],
    ['agr', ['bold']],
    ['anx', ['bold']],
    ['cnv', ['bold']],
    ['dcl', ['bold']],
    ['exl', ['bold']],
    ['exl', ['bold']],
    ['fna', ['bold']],
    ['ltr', ['bold']],
    ['mnt', ['bold']],
    ['pro', ['bold']]
  ])],
  ['OJ', new Map([
    ['enc', ['bold']],
    ['art', ['bold']],
    ['cpt', ['italic']],
    ['prt', ['italic']],
    ['tis', ['italic']],
    ['agr', ['bold']],
    ['anx', ['bold']],
    ['cnv', ['bold']],
    ['dcl', ['bold']],
    ['exl', ['bold']],
    ['exl', ['bold']],
    ['fna', ['bold']],
    ['ltr', ['bold']],
    ['mnt', ['bold']],
    ['pro', ['bold']]
  ])]
]);

//Subdivisions using roman numbering
const subdivisionProperties = new Map([
		// noNumbering : hide element numbering from toc label
    // noType : hide Type & numberinf from toc label
	['pbl', ['noNumbering']],
    ['cit', ['noNumbering']],
    ['fnp', ['noNumbering']],
    ['enc', ['noNumbering']],
    ['tit', ['noType']]
]);

//method for getting filters from BO
function transformFiltersFromBO(){

    const conslegFilters = tocEliFiltersConsleg;
    const ojFilters = tocEliFiltersOj;

    if(conslegFilters){
        filtersByTypeOfDocument.set('CONSLEG', conslegFilters.split(','));
    }

    if(ojFilters){
        filtersByTypeOfDocument.set('OJ', ojFilters.split(','));
    }

}

//method for translating the given ELI subdivision code
function translateEliSubdivisions(eliSubdivisionCode = '', suffix) {

  let translatedEliSubdivision = (eliSubdivisionLabels && eliSubdivisionLabels[eliSubdivisionCode.toUpperCase()]) || eliSubdivisionCode;

  let defaultLabels = [translatedEliSubdivision];

  const propsByType = subdivisionProperties.get(eliSubdivisionCode) || [];

  if (propsByType.includes('noType')) {
    return [];
  }

  if (suffix && !propsByType.includes('noNumbering')) {
  	defaultLabels.push(suffix);
  }

  	return defaultLabels.join(' ');
}

//method for getting css style per type of document and ELI subdivision
function getCssStyles(eliSubdivision, typeOfDocument) {
  const stylesByType = stylesByTypeOfDocument.get(typeOfDocument);
  const styles = stylesByType && stylesByType.get(eliSubdivision) || [];
  const boldStyle = styles.includes('bold') ? 'font-weight:bold;' : '';
  const italicStyle = styles.includes('italic') ? 'font-style:italic;' : '';
  const uppercaseStyle = styles.includes('uppercase') ? 'text-transform:uppercase;' : '';

  return styles.length > 0 ? `${boldStyle}${italicStyle}${uppercaseStyle}` : '';
}

// method for pruning label length if too long, and breaking at word-breaks
function pruneTocLabel(labelContent) {
	if (labelContent.length > tocEliTitleMaxLength) {
    let labelPruned = labelContent.substring(0, tocEliTitleMaxLength);
		return labelPruned.replace(/\s+\S*$/, " &hellip;");
  }
  return labelContent;
}

//method for checking ELI subdivision code
function checkExactMatchOfEliSubdivisionCode(id = '') {
  return eliSubdivisions.includes(id.toUpperCase());
}

//method for extracting ELI subdivision code
function extractEliSubdivisionCode(id = '') {
  return id.substring(0, 3).toLowerCase();
}

//method for checking if the element starts with an ELI subdivision code
function startsWithEliSubdivision(element = ''){
    if(element.length < 3){
  	    return false;
    }

    let eli = extractEliSubdivisionCode(element);

    return checkExactMatchOfEliSubdivisionCode(eli);
}

//method for extracting value after underscore
function extractValueAfterUnderscore(id = '') {
  let idx = id.indexOf("_");
  return idx === -1 ? '' : id.substring(idx + 1);
}

//method for splitting the token within a given id (e.g. tis_2.cpt_10.sct)
function splitId(id = '') {
  return id.split('.');
}

//method for checking a provided ID attribute
function checkIdAttributeValue(id = '') {
  let parts = splitId(id) || [];
  let numberOfToken = parts.length;
  let numberOfTokenApproved = 0;

  parts.forEach(elementId => {
    let eliSubdivision = extractEliSubdivisionCode(elementId);
    let checkMatch = checkExactMatchOfEliSubdivisionCode(eliSubdivision);
    let valueAfterUnderScore = extractValueAfterUnderscore(elementId);
    // manage edge case of accepting only root-level "tit" subs
    let allowTit = eliSubdivision != 'tit' || parts.length == 1;
    //let allowCit = eliSubdivision != 'cit' || parseInt(valueAfterUnderScore) == 1;
    if (checkMatch && allowTit) {
    	++numberOfTokenApproved;
    }
  });

  let lastToken = parts[numberOfToken - 1];
  let lastTokenEliSubdivision = extractEliSubdivisionCode(lastToken);

  let valueAfterUnderscore = extractValueAfterUnderscore(lastToken);
  let isApproved = numberOfToken === numberOfTokenApproved;

  //Take into account only FIRST citation
  if (lastTokenEliSubdivision == 'cit' && parseInt(valueAfterUnderscore) > 1) {
      isApproved = false;
  }

  return {
  		id: id,
        numberOfToken: numberOfToken,
        numberOfTokenApproved: numberOfTokenApproved,
        approved: isApproved,
        lastToken: lastToken,
        lastTokenEliSubdivision: lastTokenEliSubdivision,
        lastTokenValueAfterUnderscore: valueAfterUnderscore
  }

}

//method for creating the data structure used to generate the TOC
function traverseEliStructure(rootId = 'document1', typeOfDocument = 'CONSLEG') {

  //we get the applied filters
  const filters = filtersByTypeOfDocument.get(typeOfDocument) || [];

  //init of the returned data
  let data = {
    id: 'root',
    labels: [],
    children: []
  };

  //we get the HTML element to be parsed
  var element = document.getElementById(rootId);

  function getLabelArray(node) {

      if (node.classList.contains('eli-main-title')) {
      	// Main title element, edge case (can be multiple per document, ie. one per eli-container)
        // Assume it is the first child of this node
        var label = [node.firstElementChild];
      } else {
        //Only check within direct node children
      	var label = Array.from(node.children).filter(child => child.classList.contains('eli-title'));
      }
      return label.length > 0 ? [cleanTextLabel(label[0].innerHTML)] : []
  }

  	//method for cleaning label=
  function cleanTextLabel(text = '') {
  		//remove any html tag with regex
  		var linkRegex = /<[^>]*>/gm;
  		var newText = text.replace(linkRegex, "");
  		newText = newText.replace("◄", "");
  		return newText.trim();
  }

  //method for detecting an ELI structural subdivision
  function detectEliStructuralSubdivision(node, data) {
    //we get the element ID
    var id = node.id;
    //we get the element tag name
    var tagName = node.tagName;
    //data object to be passed from parent to children
    var objToBePassed = data;

    //only DIV with id attribute are considered
    if (id && tagName === 'DIV') {

      //we check if the ID value match the required pattern and return the ELI structural subdivisions information.
      var element = checkIdAttributeValue(id);

      //when the element is (1) approved and (2) structural and (3) not in the filters list, we insert the new structural element in the current data structural element.
      if (element && element.approved && !filters.includes(element.lastTokenEliSubdivision)) {
        //init new structural element
        var obj = {
          id: id,
          eliSubdivision: element.lastTokenEliSubdivision,
          suffix: element.lastTokenValueAfterUnderscore,
          labels: getLabelArray(node),
          cssStyles: getCssStyles(element.lastTokenEliSubdivision, typeOfDocument),
          children: []
        };
        //we'll return the new structural element for further processing
        objToBePassed = obj;
        //add the new structural element to the current data structural element
        data.children.push(obj);
      }

    }

    //return data structural element
    return objToBePassed;
  }

  //recursive method for checking all nodes of the parse document
  function testNodes(node, test, data) {

    var objToBePassed = detectEliStructuralSubdivision(node, data);

    node = node.firstChild;
    while (node) {
      testNodes(node, test, objToBePassed);
      node = node.nextSibling;
    }

  }

  //init recursive method
  testNodes(element, detectEliStructuralSubdivision, data);
  return data;
}

//method for generating li element
function generateLiElement(item = {}, topLabel = '', topLevelId = '') {
  let hasChildren = item.children && item.children.length > 0;
  let eliSubdivision = item.eliSubdivision;
  let suffix = item.suffix;
  let labels = item.labels || [];
  let cssStyles = item.cssStyles ? ` style="${item.cssStyles}"` : '';
  //if we have labels we take those otherwise we take the eli subdivision code and suffix
  let labelElement = labels.length > 0 ? `<span class='toc-eli-label'>${pruneTocLabel(labels[0])}</span>` : [];
  let textElement = [translateEliSubdivisions(eliSubdivision, suffix), labelElement].flat().join(' - ');
  let hrefElement = topLevelId ? topLevelId : item.id;
  let innerElement = hasChildren ? `<i class="glyphicon glyphicon-chevron-right"></i>${textElement}` : textElement;

  return `<li><a${cssStyles} href="#${hrefElement}">${innerElement}</a>`;
}

//method for processing the TOC based on the generated data structure
function processEliSubdivisionsDataStructure(data = [], topLabel, titleCount = 0, selector = '') {

  let temp = '';

  for (let item of data) {

        //MAIN DOCUMENT TITLE = TIT
    	if(item.id === 'tit' && titleCount === 0){
        	//generate li element with new ID (resolved issue with TIT duplicate IDs)
            let topLevelId = `${selector}-topLink`;
        	temp += generateLiElement(item, topLabel, topLevelId);
            changeTopLevelIdForEli(selector);
            //increment title counter
            ++titleCount;
        }

        //OTHER STRUCTURAL ELI SUBDIVISIONS
        if(item.id !== 'tit'){
        	//generate li element
        	temp += generateLiElement(item, topLabel);
        }

    //when children are available
    if (item.children && item.children.length > 0) {
      //recursively generate children
      let innerElements = processEliSubdivisionsDataStructure(item.children, topLabel, titleCount, selector);
      temp += `<ul class="nav collapse">${innerElements}</ul>`;
    }

    //close li element
    temp += '</li>';

  }

  return temp;

}

//method for generating the TOC based on the data structure.
function generateTOCForEliSubdivision(selector, documentType, topLabel) {
    //we count the number of title included. Should be for a document not greater than 1
    let titleCount = 0;

    //get filters from BO
    transformFiltersFromBO();

    //create data structure
    let data = traverseEliStructure(selector, documentType);

    //we use in case of only tit elements the old algo
    if(hasOnlyTitElements(data)){
          return undefined;
    }

    //we generate TOC only if at least one subdivision exists
    if (data.children && data.children.length > 0) {

          let links = processEliSubdivisionsDataStructure(data.children, topLabel, titleCount, selector);

          return `<ul id="TOC_${selector}" class="toc-eli-subdivisions nav">${links}</ul>`;

    }

    return undefined;

}

//method for checking if we have only tit elements
function hasOnlyTitElements(data){

	let titCounter = 0;
    let size = 0;

	if(data.children && data.children.length>0){
        size = data.children.length;

        for(const item of data.children){
            if(item.id === 'tit'){
                ++titCounter;
            }
        }
    }

    return size === titCounter;
}

//method for changing ID of title
function changeTopLevelIdForEli(selector){
    //TODO ADJUST BY ADDING .topLink in class
	$(`#${selector} div[id=tit]`).first().attr("id", `${selector}-topLink`);
}

//method used for expand/collapse node in TOC
function toggleTocEliMenu(selector) {
  $(`#TOC_${selector}`).on('click', 'li > a', function(e) {
  		//find parent li element
  		let listItem = $(this).parent('li');
  		//check if children
  		if (listItem.children('ul').length > 0) {
  			//get chevron of the active link
  			let chevron = $(this).find('.glyphicon');
  			//stop event propagation to prevent toggling inner <li> elements
  			e.stopPropagation();
  			//toggle ul
  			listItem.children('ul').toggle();
  			//toggle chevron
  			chevron.toggleClass('glyphicon-chevron-down glyphicon-chevron-right');
  		}

  	});
}

//method for generating the top link
function appendMainTitleAsTopLink(topLabel = ''){
    $('.eli-main-title').each(function(){
            let id = this.id;
            let link = `<a href="#${id}">${topLabel}</a>`;
            $(".topBar").append(link);
    });
}

//Listener for adding active class on link
function tocListenerOnActiveLink() {
  $('.toc-eli-subdivisions a').on('click', function(e) {
        e.preventDefault();
  		$('.toc-eli-subdivisions a').removeClass("active");
  		$(this).addClass("active");
  });
}

function topButtonEliListener(){
    if(tocEliSubdivisionsEnable){
        $('.topBar a').on('click', function(e) {
            console.log("TopBar clicked");
            //scroll the toc to the top
            $('.toc-sidebar').animate({scrollTop: 0}, 'fast');
            $('.toc-eli-subdivisions a').removeClass("active");
            $(this).addClass("active");
        });
    }
}

//EURLEXNEW-4576

//method used for appending a hash when eliSubdivisionsWithDots is available
function eliResolutionResolver(relativeRequestUrl = '') {

    //split the ELI subdivisions dots to an array
    let arrayEliSubdivisions = retrieveEliResolutionPathFromRelativeRequestUrl(relativeRequestUrl);

    console.log(`The used ELI Subdivisions are : ${arrayEliSubdivisions}`)

    //we get array from first level available
    let arrayFromFirstLevelFound = constructEliPathFromFirstLevel(arrayEliSubdivisions);

    let size = arrayFromFirstLevelFound.length;

    let id = traverseEliResolutionPath(arrayFromFirstLevelFound, size - 1, size - 1);

    if (id) {
      console.log(`The document will be scrolled to ${id}`);
      scrollToProvidedId(id);
    }

}

//method for traversing ELI resolution path
function traverseEliResolutionPath(arrayFromFirstLevel, from, to) {

  let id = generateIDFromEliSubdivisions(arrayFromFirstLevel, from, to);

  if (hasElementWithId(id)) {
    return id;
  }

  if (from > 0 && to > 0) {
    return traverseEliResolutionPath(arrayFromFirstLevel, from - 1, to);
  }

  if (from === 0 && to > 0) {
    return traverseEliResolutionPath(arrayFromFirstLevel, from, to - 1);
  }

}

//method for generating an ID dynamically from array
function generateIDFromEliSubdivisions(arrayFromFirstLevel, from, to) {
  let temp = arrayFromFirstLevel.slice(from, to + 1);
  return temp.join('.');
}

//method for generating the Eli path from first level
function constructEliPathFromFirstLevel(array = []) {
  let size = array.length;
  let lastIdx = size > 0 ? size - 1 : size;
  let firstLevelIdx = -1;

  for (let i = lastIdx; i >= 0; i--) {
    if (hasElementWithId(array[i])) {
      firstLevelIdx = i;
      break;
    }
  }

  return firstLevelIdx !== -1 ? array.slice(firstLevelIdx) : [];
}

//check if element with ID exists
function hasElementWithId(id) {
  return document.getElementById(id);
}

//method for retrieving the ELI subdivisions within the provided path
function retrieveEliResolutionPathFromRelativeRequestUrl(relativeRequestUrl = ''){
	let pathElements = relativeRequestUrl.split('/');
    let values = [];

    for(const element of pathElements){
  	    if(startsWithEliSubdivision(element)){
    	    values.push(element);
        }
    }

    return values;
}

//method used to scroll to the provided ID
function scrollToProvidedId(id){

    let sanitizedId = sanitizeHtml(id);

    let idToScrollTo = sanitizedId === 'tit' ? 'textTabContent' : sanitizedId;

    if($(`#${idToScrollTo}`).length){
        $('html, body').animate({
            scrollTop: $(`#${idToScrollTo}`).offset().top
        }, 'fast');
    }
}

//EURLEXNEW-4571: method to reset the scroll to the URL anchor position.
function scrollToCurrentUrlAnchor(){
    let hash = sanitizeHtml($(location).attr('hash'));
    if(hash){
        scrollToProvidedId(hash.substring(1));
    }
}

function handleActiveOnPageLoad(){

    let hash = sanitizeHtml($(location).attr('hash'));

    if(hash && tocEliSubdivisionsEnable){
        const menuItem = $(`a[href="${hash}"]`);
        if (menuItem.length){

            //remove all active class
            $('.toc-eli-subdivisions a').removeClass("active");

            //add active class to the related menu item
            menuItem.addClass('active');

            //expand all parent menus
            menuItem.parents('ul').each(function(){
                $(this).show();
                $(this).siblings('a').find('i').removeClass('glyphicon-chevron-right').addClass('glyphicon-chevron-down');
            });

            //scroll into view
            menuItem[0].scrollIntoView({behavior:'smooth',block:'center'})

        }
    }
}