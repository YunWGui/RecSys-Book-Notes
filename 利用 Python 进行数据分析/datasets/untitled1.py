# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:10:53 2019

@author: SHI YI
"""

# -*- coding:utf-8*-

import os
import os.path
from PyPDF2 import PdfFileReader, PdfFileWriter
import time
import glob



def getFileName(filepath):

    file_list = glob.glob("*.pdf")
    # 默认安装字典序排序，也可以安装自定义的方式排序
    # file_list.sort()
    return file_list


##########################合并同一个文件夹下所有PDF文件########################
def MergePDF(filepath, outfile):
    output = PdfFileWriter()
    outputPages = 0
    pdf_fileName = getFileName(filepath)
    for each_file in pdf_fileName:
        print("adding %s" % each_file)
        # 读取源pdf文件
        input = PdfFileReader(open(each_file, "rb"))

        # 如果pdf文件已经加密，必须首先解密才能使用pyPdf
        if input.isEncrypted == True:
            input.decrypt("map")

        # print(each_file[:-4])

        # 获得源pdf文件中页面总数
        pageCount = input.getNumPages()
        outputPages += pageCount
        print("%s has %d pages" % (each_file, pageCount))

        # 分别将page添加到输出output中
        for iPage in range(pageCount):
            output.addPage(input.getPage(iPage))

        # 添加书签
        output.addBookmark(
            title=each_file[:-3], pagenum=outputPages - pageCount)

    print("All Pages Number: " + str(outputPages))
    # 最后写pdf文件
    outputStream = open(filepath + outfile, "wb")
    output.write(outputStream)
    outputStream.close()
    print("finished")


if __name__ == '__main__':
    time1 = time.time()
    file_dir = '.'
    out = u"test.pdf"
    MergePDF(file_dir, out)
    time2 = time.time()
    print(u'总共耗时： %.4f s' % (time2 - time1))
