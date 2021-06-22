# fft 和 fftshift

这里主要说明一点, 各个语言 (matlab 或 python) 数值算法中, fft 之后的频率是从 0 开始的, 而 fftshift 的作用就是将 fft 所得频率将 0 频率移动到中间.



在matlab中，经过fft变换后，数据的频率范围是从[0,fs]排列的。而一般，我们在画图或者讨论的时候，是从[-fs/2,fs/2]的范围进行分析。因此，需要将经过fft变换后的图像的[fs/2,fs]部分移动到[-fs/2,0]这个范围内。

而fftshift就是完成这个功能。通常，如果想得到所见的中间是0频的图像，经过fft变换后，都要再经过fftshift这个过程。



可以参考 matlab 官方文档 https://ww2.mathworks.cn/help/matlab/ref/fftshift.html

