require 'sysutils'

function ParsingShoes(m)
    local path, filename = splitfilename(m[2])
    local category = m[3]
    local box = {x=m[4],y=m[5],width=m[6],height=m[7]}
    local properties = torch.Tensor(17):zero()

    local gender = m[8]
    if gender == '성인남성' then
        properties[1] = 1
    elseif gender == '성인여성' then
        properties[1] = 2
    elseif gender == '아동남성' then
        properties[1] = 3
    elseif gender == '아동여성' then
        properties[1] = 4
    end

    local material = m[9]
    if material == '가죽' then
        properties[2] = 1
    elseif material == '천' then
        properties[2] = 2
    elseif material == '고무' then
        properties[2] = 3
    end

    local ankle = m[10]
    if ankle == '복숭아뼈아래' then
        properties[3] = 1
    elseif ankle == '복숭아뼈위' then
        properties[3] = 2
    elseif ankle == '종아리' then
        properties[3] = 3
    elseif ankle == '무릎' then
        properties[3] = 4
    end

    local heels = m[11]
    if heels == '로우' then
        properties[4] = 1
    elseif heels == '미드' then
        properties[4] = 2
    elseif heels == '하이' then
        properties[4] = 3
    end

    local pattern = m[12]
    if pattern == '무지' then
        properties[5] = 1
    elseif pattern == '브랜드 로고' then
        properties[5] = 2
    elseif pattern == '스트라이프' then
        properties[5] = 3
    elseif pattern == '도트' then
        properties[5] = 4
    elseif pattern == '호피/지브라' then
        properties[5] = 5
    elseif pattern == '도형' then
        properties[5] = 6
    elseif pattern == '이니셜' then
        properties[5] = 7
    elseif pattern == '밀리터리' then
        properties[5] = 8
    elseif pattern == '배색' then
        properties[5] = 9
    elseif pattern == '꽃무늬' then
        properties[5] = 10
    elseif pattern == '뱀피' then
        properties[5] = 11
    elseif pattern == '체크' then
        properties[5] = 12
    elseif pattern == '일러스트' then
        properties[5] = 13
    end

    -- forefood
    if m[13] == 'O' then
        properties[6] = 1
    end
    -- 끈
    if m[14] == 'O' then
        properties[7] = 1
    end
    -- 지퍼
    if m[15] == 'O' then
        properties[8] = 1
    end
    -- 앞트임
    if m[16] == 'O' then
        properties[9] = 1
    end
    -- 뒷트임
    if m[17] == 'O' then
        properties[10] = 1
    end
    -- 메쉬
    if m[18] == 'O' then
        properties[11] = 1
    end
    -- 리본장식
    if m[19] == 'O' then
        properties[12] = 1
    end
    -- 단추장식
    if m[20] == 'O' then
        properties[13] = 1
    end
    -- 버클
    if m[21] == 'O' then
        properties[14] = 1
    end
    -- 클립
    if m[22] == 'O' then
        properties[15] = 1
    end
    -- 벨트
    if m[23] == 'O' then
        properties[16] = 1
    end
    -- 비즈/징
    if m[24] == 'O' then
        properties[17] = 1
    end

    return filename, box, properties
end

