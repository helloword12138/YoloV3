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
    J = jQuery;

    var Authc = {
        isAuthced: function () {
            return (typeof(window.app.LOGIN_TOKEN) != 'undefined' && window.app.LOGIN_TOKEN.length > 0);
        },
        showLogin : function () {
            var that = this;
            $('#loginalert').modal();

            $('#alt_login').unbind().click(function () {
                that.doPostLogin();
            });
        },
        doPostLogin: function () {
            var un = $('#alt_un').val();
            var pw = $('#alt_pw').val();
            jQuery.post(app.base + '/api/login.json', {'username': un, 'password': pw}, function (ret) {
                if (ret && ret.code == 0) {
                    //$('#loginalert').modal('hide');
                    //window.app.login = ret.data.id;
                    window.location.reload();
                } else {
                    $('#login_warning').text(ret.message).show();
                }
            });
        }
    };

    module.exports = Authc;
});