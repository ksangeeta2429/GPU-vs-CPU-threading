#comment
FLAG=-O -c -w 
CC=nvcc

all:lab4p1 lab4p2 lab4p3

lab4p1:kumari_sangeeta_lab4p1.o
	$(CC) -o $@ kumari_sangeeta_lab4p1.o

kumari_sangeeta_lab4p1.o:kumari_sangeeta_lab4p1.cu
	$(CC) $(FLAG) kumari_sangeeta_lab4p1.cu

lab4p2:kumari_sangeeta_lab4p2.o
	$(CC) -o $@ kumari_sangeeta_lab4p2.o

kumari_sangeeta_lab4p2.o:kumari_sangeeta_lab4p2.cu
	$(CC) $(FLAG) kumari_sangeeta_lab4p2.cu

lab4p3:kumari_sangeeta_lab4p3.o
	$(CC) -o $@ kumari_sangeeta_lab4p3.o

kumari_sangeeta_lab4p3.o:kumari_sangeeta_lab4p3.cu
	$(CC) $(FLAG) kumari_sangeeta_lab4p3.cu
    
clean:
	rm ./*.o
	rm ./lab4p1
	rm ./lab4p2
	rm ./lab4p3