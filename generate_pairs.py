#-*- coding: utf-8 -*-

filename = 'fashion_pair.csv'
outfilename = './html/IR/pair_lists.html'
image_baseurl = "http://175.126.56.112/october_11st/"
f = open(filename, 'rt')

meta = dict()

count = 0
for line in f.readlines():
    line = line.rstrip()
    items = line.split(",")
    anchor = items[1]
    pair = items[2]
    p_or_n = items[3]
    if not anchor in meta:
        meta[anchor] = {"P":[], "N":[]}
    if p_or_n is '1':
        meta[anchor]["P"].append( pair )
    else:
        meta[anchor]["N"].append( pair )
    count += 1

f.close()

print "len(meta): {}".format(len(meta))
print "count: {}".format(count)

fout = open(outfilename, 'wt')
fout.write("""<html>
        <head>
        <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
        <script src="http://code.jquery.com/jquery-latest.js"></script>
        <style>
        .img_anchor {
            border: 1px solid black;
            width: 128px;
            height: 128px;
        }
        .img_positive {
            border: 1px solid red;
            width: 128px;
        }
        .img_negative {
            border: 1px solid blue;
            width: 128px;
        }

        table {
            table-layout: fixed;
            width: 100%;
        }
        th.positive, th.negative {
            border: 1px solid black;
            width: 40%;
        }
        td.positive, td.negative {
            border: 1px solid black;
            width: 40%;
        }
        </style>
        </head>
        <body>
        <table>
        <tr>
            <th>Anchor Image</th>
            <th class="positive">Positive Images</th>
            <th class="negative">Negative Images</th>
        </tr>
""")

itemcount = 0
outcount = 0
for key, val in meta.iteritems():
    if itemcount % 50 == 0:
        line = "<tr><td class='td_anchor'><img class='img_anchor' src='{}{}'></td>".format(image_baseurl, key)
        line += "<td class='positive'>"
        for img_url in val["P"]:
            line += "<img class='img_positive' src='{}{}'>".format(image_baseurl, img_url)
        line += "</td><td class='negative'>"
        for img_url in val["N"]:
            line += "<img class='img_negative' src='{}{}'>".format(image_baseurl, img_url)
        line += "</td></tr>\n"
        fout.write(line)
        outcount += 1
    itemcount += 1
fout.write('</table></body></html>\n')
fout.close()

print ("output lines:{}".format(outcount))


