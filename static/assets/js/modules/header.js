define(function(require, exports, module) {
	//require('jquery');
	$(function () {
		$('a[nav]').each(function(){  
	        $this = $(this);
	        if($this[0].href == String(window.location)){  
	            $this.closest('li').addClass("active");
	        }  
	    });
	});
});