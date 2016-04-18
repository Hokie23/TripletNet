function splitfilename(strfilename)
    -- Returns the Path, Filename, and Extension as 3 values
    return string.match(strfilename, "^(.+)/(.+)$")
end
