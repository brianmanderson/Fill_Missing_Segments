## If you find this code useful, please provide a reference to my github page for others www.github.com/brianmanderson , thank you!

This is for filling in pixels that don't have a class associated with them, by default it uses the MauererDistanceMap in SimpleITK

    from Fill_In_Segments_sitk import Fill_Missing_Segments
    Fill_Segments = Fill_Missing_Segments()
    liver = numpy_image # Of shape [# images, 512, 512], dtype='int'
    pred = numpy_image # Of shape [# images, 512, 512, # classes+1], dtype='int'
    filled = Fill_Segments.make_distance_map(pred,liver)
