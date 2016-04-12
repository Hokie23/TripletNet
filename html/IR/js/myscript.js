String.prototype.hash = function() {
  var self = this, range = Array(this.length);
  for(var i = 0; i < this.length; i++) {
    range[i] = i;
  }
  return Array.prototype.map.call(range, function(i) {
    return self.charCodeAt(i).toString(16);
  }).join('');
}

Function.prototype.Async = function(params, cb){
      var function_context = this;
      setTimeout(function(){
          var val = function_context.apply(undefined, params); 
          if(cb) cb(val);
      }, 0);
}


function imageoverlay(e) 
{
    var div_width = 600;
    //img_x = $(this).parent().offset().left + $(this).width();
    //img_y = $(this).parent().offset().top - $(this).height();
    img_x = ($(this).offset().left + $(this).width() - $(this).parent().offset().left);
    img_y = ($(this).offset().top - $(this).parent().offset().top);
    var wr = Math.min(1.0, 300.0 / this.naturalWidth);
    var hr = Math.min(1.0, 300.0 / this.naturalHeight);
    var ratio = Math.min(wr, hr);
    var h = this.naturalHeight * ratio;
    var w = this.naturalWidth * ratio;
    console.log(h + " " + w + " " + ratio);
    console.log(this.naturalHeight + " " + this.naturalWidth);
//    if ( img_y + h > $(document).height() ) {
//        img_y = $(document).height() - h - 80;
//    }
//    else {
//
//    }
//    if ( img_x + w > $(document).width() ) {
//        img_x -= img_x + w - $(document).width() + $(document).width() - img_x + $(this).width();
//        img_x -= 20;
//    }
//    else {
//        img_x += 20;
//    }
//

    if( $(this).offset().left + div_width > $(document).width() ) {
        img_x = ($(this).offset().left - div_width - $(this).parent().offset().left);
    }
    console.log("event.data:" + e.data);
    console.log("x:" + img_x + ", y:" + img_y);
    imageurl = e.data.imageurl;
    div_tag = $('<div>').css({position: 'absolute', width: "600px", "max-width":"600px", left: img_x + "px", top: img_y + "px", "z-index": 1});
    //div_tag = $('<div>').css({position: 'relative', left: "20px", top: "-20px"});
    div_tag.attr('id', 'img_div0404');
    img_tag = $('<img>');
    img_tag.attr('src', this.src);
    img_tag.attr('style', 'border: 4px solid; border-color: #f00; max-width: 300px; max-height: 300px;');
    div_tag.append(img_tag);
    org_img_tag = $('<img>');
    org_img_tag.attr('src', imageurl);
    org_img_tag.attr('style', 'border: 4px solid; border-color: #f00; max-width: 300px; max-height: 300px;');
    div_tag.append(org_img_tag);
    $(this).parent().append(div_tag);
    //$(this).append(div_tag);
      
    $(this).mouseout(function() { $("#img_div0404").remove(); });
    //div_tag.mouseout(function() { $("#img_div0404").remove(); });
}
