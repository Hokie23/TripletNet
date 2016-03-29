#!/bin/bash

#url="https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcQitJe4u3WnxlMT_Ltn37-YqcE3lg9KvCCFRlmfvIXrgaY6_lKO"
url="http://www.teetreedesigns.co.uk/Images/GarmentImages/RedTShirt.jpg"
curl http://localhost:8080/query -d $url
