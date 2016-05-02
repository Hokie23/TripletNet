local debugger = require 'fb.debugger'
function precision_recall( conf, label )
    local num_correct = 0 -- true positives
    local threshold, index = conf:sort(true)
    local n = threshold:numel()
    local correct = label:index(1,index):eq(1):float()
    local relevant_element = correct:sum()
    local recall = torch.zeros(n)
    local precision = torch.zeros(n)
    local ap = 0.0

    --debugger.enter()
    for i = 1,n do
        --compute precision
        num_positive = i        -- true and false positive(selective elements), retrived documents
        num_correct = num_correct + correct[i]
        if num_positive ~= 0 then
            precision[i] = num_correct / num_positive;
        else
            precision[i] = 0;
        end

        if correct[i] == 0 then
            ap = ap + precision[i]
        end

        --compute recall
        recall[i] = num_correct / relevant_element
    end

    if relevant_element > 0 then
        ap = ap / relevant_element
    end

    return recall, precision, ap, threshold
end
