function test(arr) {
    var len = arr.length/2;
    var rst =[];
    for(var i =0;i<len;i++){
        rst[i] = arr[2*i+1] - arr[2*i]
    }
    return rst;
}