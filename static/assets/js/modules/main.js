/*
+--------------------------------------------------------------------------
|   Mblog [#RELEASE_VERSION#]
|   ========================================
|   Copyright (c) 2014, 2015 mtons. All Rights Reserved
|   http://www.mtons.com
|
+---------------------------------------------------------------------------
*/

define(function(require, exports, module) {
    var plugins = require('plugins');
	var Authc = require('authc');

    //require('bootstrap');

    var wpexLocalize = {
    		"mobileMenuOpen" : "Click here to navigate",
    		"mobileMenuClosed" : "Close navigation",
    		"isOriginLeft" : "1"
    	};
    
    // 图片懒加载
    var imagesLazyload = function () {
    	require.async('lazyload', function () {
    		$("img").lazyload({
	   	   		 placeholder: app.base + '/assets/images/spinner.gif',
	   	   		 effect: "fadeIn"
	   	   	});
        });
    }
    
    // 加载图片灯箱
    var initLightbox = function () {
    	require.async('baguetteBox', function () {
    		baguetteBox.run('.thumbs', {
        		animation: 'slideIn'
        	});
        });
    };
    
    // 返回顶部
    var backToTop = function () {
    	var $window = $(window);
    	$scrollTopLink = $( 'a.site-scroll-top' );
		$window.scroll(function () {
			if ($(this).scrollTop() > 100) {
				$scrollTopLink.fadeIn();
			} else {
				$scrollTopLink.fadeOut();
			}
		});		
		$scrollTopLink.on('click', function() {
			$( 'html, body' ).animate({scrollTop:0}, 400);
			return false;
		} );
    }
    
	// 瀑布流
    var masonry = function () {
    	// Masonry
		var $container = $('.masonry-grid');
		$container.imagesLoaded(function(){
			// FlexSlider run after images are loaded
			$container.masonry({
				itemSelector: '.masonry-entry',
				transitionDuration: '0.3s',
				isOriginLeft: wpexLocalize.isOriginLeft
			});
		});

		// Infinite scroll
		var $container = $('.masonry-grid');
		$container.infinitescroll( {
			loading: {
				msg: null,
				finishedMsg : null,
				msgText : null,
				msgText: '<div class="infinite-scroll-loader"><i class="fa fa-spinner fa-spin"></i></div>',
			},
				navSelector  : 'div.page-jump',
				nextSelector : 'div.page-jump div.older-posts a',
				itemSelector : '.masonry-entry',
			},
			// trigger Masonry as a callback
			function( newElements ) {
				// hide new items while they are loading
				var $newElems = $( newElements ).css({ opacity: 0 });
				// ensure that images load before adding to masonry layout
				$newElems.imagesLoaded(function(){
					// show elems now they're ready
					$newElems.animate({ opacity: 1 });
					$container.masonry( 'appended', $newElems, true );
					// Self hosted audio and video
			});
		});
    }


	// 绑定按钮事件
	var bindClickEvent = function () {
		// Favor
		$('a[rel=favor]').click(function () {
			var id = $(this).attr('data-id');

			if (!Authc.isAuthced()) {
				Authc.showLogin();
				return false;
			}

			if (parseInt(id) > 0) {
				jQuery.getJSON(app.base +'/account/favor', {'id': id}, function (ret) {
					
					if (ret.code == 2) {
						var favors = $('#favors').text();
						$('#favors').text(parseInt(favors) + 1);
					}
					if (ret.code == 1) {
						layer.msg("您已经喜欢过此文章", {icon: 5});
					}
					if (ret.code == 0) {
						layer.msg("您尚未登录，请登录后点击", {icon: 5});
					}
				});
			}
		});

		// Follow
		$('a[rel=follow]').click(function () {
			var that = $(this);
			var id = that.attr('data-id');

			if (!Authc.isAuthced()) {
				Authc.showLogin();
				return false;
			}

			if (parseInt(id) > 0) {
				jQuery.post(app.base +'/account/follow', {'id': id}, function (ret) {
					if (ret.code==0) {
						layer.msg("未登录",function(){});
					}
					if (ret.code==1) {
						that.text("已关注").css('border-color','green');
					}
					if (ret.code==3) {
						
					}
				});
			}
		});

		$('a[rel=follow]').each(function () {
			var that = $(this);
			var id = that.attr('data-id');

			if (parseInt(id) > 0) {
				jQuery.post(app.base +'/account/follow/check/'+id, {}, function (ret) {
					//当code为0时，代表未登录，不管
					if (ret.code == 2) {
						that.text("已关注").css('border-color','green').off('click');
					}
				});
			}
		});

		//$(document).pjax('a[rel=pjax]', '#wrap', {
		//	fragment: '#wrap',
		//	timeout: 10000,
		//	maxCacheLength: 0
		//});
	}

    exports.init = function () {
    	imagesLazyload();

    	initLightbox();
    	
    	backToTop();
    	
    	masonry();

		bindClickEvent();
    }
    
});