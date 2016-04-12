var MyServerExport = {};
( function(exports) {

var Config = {
    _remote_server: "10.202.35.87:8080"
};

Config.GetRemoteServer = function() {
    return this._remote_server;
};


var MyServer = function() {
    this._remote_server = ""
    this._session_id = "";
    this.isConnected = false;
};

MyServer.prototype.OnCallBack = function(service_name, event_type, result, textStatus, extra_param) { 
}

MyServer.prototype.SetCallBack = function( NewCallBack ) {
    this.OnCallBack = NewCallBack;
}

MyServer.prototype.RemoteServer = function() {
    console.log( this._remote_server )
    return this._remote_server;
}

MyServer.prototype._internalRequest = function(service_uri, service_name, senddata) {
    var _this = this;
    var result = $.ajax( {url: "http://"+ _this.RemoteServer() + service_uri,
            dateType: 'json',
            type: 'POST',
            crossDomain: true,
            xhrFields: {withCredentials: false},
            data: JSON.stringify(senddata),
            async: true,
            cache: false,
            success: function(result, textStatus, jqXHR ) {
                console.log( result );
                _this.OnCallBack(service_name, "success", result, textStatus, {jqXHR: jqXHR, error: null});
            },
            error:function(request,textStatus,error){
                console.log("code:"+request.status+"\n"+"message:"+request.responseText+"\n"+"error:"+error);
                _this.OnCallBack(service_name, "error", result, textStatus, {jqXHR: null, error: error});
            }
        });
    console.log("ajax finished");
}

MyServer.prototype.RetrievalImage = function(image_url, category) {
    var senddata = {
        image_url: image_url,
        category: category
    };

    this._internalRequest( "/query", "retrievalimage", senddata );
}


exports.MyServer = MyServer;
})(MyServerExport);

var MyServer = MyServerExport.MyServer;
