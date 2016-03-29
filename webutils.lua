local curl = require('luacurl')
local http = require('socket.http')
local ltn12 = require('ltn12')

function HTTPRequest(url)
    local body = {}
    local client, code, headers, status = http.request( {
        method='GET',
        url=url,
        sink=ltn12.sink.table(body)
        })
    
    return table.concat(body), headers, status, code
end

function DownloadFileFromURL(url)
    local c = curl.new()
    print( "url", url)
    c:setopt(curl.OPT_URL, url)
    local t = {}
    local h = {}

    c:setopt( curl.OPT_WRITEFUNCTION, function(param, buf)
                table.insert(t, buf)
                return #buf
            end)
    c:setopt( curl.OPT_HEADERFUNCTION, function(param, buf)
                table.insert(h,buf)
                return #buf
            end)
    c:setopt( curl.OPT_PROGRESSFUNCTION, function(param, dltotal, dlnow)
            print ("%", url, dltotal, dlnow)
        end )
    c:setopt(curl.OPT_NOPROGRESS, false)
    assert(c:perform())
    c:close()
    return table.concat(t), table.concat(h)
end
