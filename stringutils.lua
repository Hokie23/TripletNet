
string.split = function(str, delim)
    local t = {}
    local tt = {}
    local s = str
    local dm = delim
    local limit_t = limit
    local ck = false or limit
    str = nil
    delim = nil
    limit = nil
    while true do
        if s == nil then break end
        local fn = function(t, s, delim)
            local idx = select(2, string.find(s, delim))
            if idx == nil then
                table.insert(t, string.sub(s, 0))
                return nil
            else
                table.insert(t, string.sub(s, 0, idx - 1))
                return string.sub(s, idx + 1)
            end
        end

        s = fn(t, s, dm);
    end
    if not ck then return t end
    for i = 1, limit_t do
        table.insert(tt, t[i])
    end
    return tt
end
