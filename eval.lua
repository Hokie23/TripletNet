
function precision_recall( conf, label )
    local num_correct = 0 -- true positives
    local threshold, index = conf:sort()
    local n = threshold:numel()
    local correct = label:index(1,index):eq(1):float()
    local relevant_element = correct:sum()
    local recall = torch.zeros(n)
    local precision = torch.zeros(n)

    for i = 1,n do
        --compute precision
        num_positive = i        -- true and false positive(selective elements), retrived documents
        num_correct = correct + correct[i]
        if num_positive ~= 0 then
            precision[i] = num_correct / num_positive;
        else
            precision[i] = 0;
        end

        --compute recall
        recall[i] = num_correct / relevant_element
    end

    return recall, precision, threshold
end
