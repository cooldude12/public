Read -p "enter the first number: "
Read -p "enter the second number: "

if ((x < y)); then
    echo "$x < $y"
elif ((x > y)); then
    echo "$y < $x"
else
    echo "$x == $y"
fi
