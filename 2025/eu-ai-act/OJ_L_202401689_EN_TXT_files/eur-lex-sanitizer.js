//To prevent a XSS attack, this method is used to sanitize any passed string using jQuery.
function sanitizeHtml(str){
    return $("<div>").text(str).html();
}