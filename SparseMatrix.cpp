#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include <omp.h>
#include <stdexcept> 
#include <thread>
#include <omp.h>
#include <algorithm>
#include <vector>

template <typename T>
class CSC_Format {
private:
    struct Node {
        T value;
        int row;
        Node* next;

        Node(int r, T v, Node* n = nullptr) : row(r), value(v), next(n) {}
    };

    Node** nodesBegin; // Array of pointers to the beginning of each row
    Node** nodesEnd; // Array of pointers to the end of each row
    int* col_pointers; // Array of row pointers
    int* col_noe;
    int rows;
    int cols;
    int nnz = 0;
    T zero = 0;

public:
    CSC_Format(int r, int c) : rows(r), cols(c) {
        nodesBegin = new Node * [cols] {};  // Initialize all pointers to nullptr
        nodesEnd = new Node * [cols] {}; // Initialize all pointers to nullptr
        col_pointers = new int[cols + 1]();
        col_noe = new int[cols]();
    }

    ~CSC_Format() {


        delete[] col_pointers;
        delete[] col_noe;

        for (int i = 0; i < cols; ++i) {
            Node* current = nodesBegin[i];
            while (current) {
                Node* next = current->next;
                delete current;
                current = next;
            }
        }
        delete[] nodesBegin;
        delete[] nodesEnd;
    }

    void addElement(int row, int col, T value) {
        Node* newNode = new Node(row, value, nullptr);

        if (!nodesBegin[col]) {
            // If the first node in the row does not exist, add the new node
            nodesBegin[col] = newNode;
            nodesEnd[col] = newNode;  // Update the nodesEnd pointer
        }
        else {
            // If the first node already exists, find the right place to insert the new node
            Node* current = nodesBegin[col];
            Node* prev = nullptr;

            while (current && row > current->row) {
                prev = current;
                current = current->next;
            }

            if (prev && prev->row == row) {
                prev->value = value;
                return;
            }

            if (!prev) {

                // New node should be the first in the col
                newNode->next = current;
                nodesBegin[col] = newNode;  // Update the nodes pointer
            }
            else {

                // Insert the new node between the previous and current nodes
                prev->next = newNode;
                newNode->next = current;

                // If current is nullptr, the newNode is the new end of the row
                if (!current) {
                    nodesEnd[col] = newNode;  // Update the nodesEnd pointer
                }
            }
        }

        col_noe[col]++;  // Increase the number of elements in the row
        nnz++;
    }


    // Za brze dodavanje kod mnozenja Forward
    void addElementForward(int row, int col, T value) {
        Node* newNode = new Node(row, value, nullptr);

        if (!nodesBegin[col]) {
            // If the first node in the row does not exist, add the new node
            nodesBegin[col] = newNode;
            nodesEnd[col] = newNode;  // Update the nodesEnd pointer
        }
        else {
            // If the first node already exists, find the right place to insert the new node
            Node* current = nodesEnd[col];
            current->next = newNode;
            nodesEnd[col] = newNode;
        }

        col_noe[col]++;  // Increase the number of elements in the row
        nnz++;
    }


    const T& GetElement(int row, int col) const {
        if ((nodesBegin[col] && row < nodesBegin[col]->row) || (nodesEnd[col] && row > nodesEnd[col]->row))
            return zero;
        Node* current = nodesBegin[col];
        while (current) {
            if (current->row == row)
                return current->value;
            if (current->row > row)
                return zero;
            current = current->next;
        }
        return zero;
    }


    int getNNZ() const {
        return nnz;
    }

    const int* getColPointers() const {
        for (int i = 1; i <= rows; ++i) {
            col_pointers[i] = col_pointers[i - 1] + col_noe[i - 1];
        }
        return col_pointers;
    }

    T* GetColumn(int col) {
        T* values = new T[rows]();               

        Node* current = nodesBegin[col];
        while (current) {
           values[current->row] = current->value;               
           current = current->next;
        }
        return values;
    }


    void printColPointers() const {
        getColPointers();
        std::cout << "Col Pointers: ";
        for (int i = 0; i <= cols; ++i) {
            std::cout << col_pointers[i] << " ";
        }
        std::cout << std::endl;
    }

    void printNodesBeginEnd() const {
        for (int i = 0; i < cols; i++) {
            auto node = nodesBegin[i];
            if (node) {
                auto end = nodesEnd[i];
                std::cout << "Pocetak reda " << i << " kolona " << node->row << " vrijednost " << node->value << " kraj reda kolona " << end->row << " vrijednost " << end->value << std::endl;
            }
        }
    }
};

/// <summary>
///
/// </summary>
/// <typeparam name="T"></typeparam>

template <typename T>
class CSR_Format {
private:
    struct Node {
        T value;
        int col;
        Node* next;

        Node(int c, T v, Node* n = nullptr) : col(c), value(v), next(n) {}
    };

    Node** nodesBegin; // Array of pointers to the beginning of each row
    Node** nodesEnd; // Array of pointers to the end of each row
    int* row_pointers; // Array of row pointers
    int* row_noe;
    int rows;
    int cols;
    int nnz = 0;
    T zero = 0;

public:
    CSR_Format(int r, int c) : rows(r), cols(c) {
        nodesBegin = new Node * [rows] {};  // Initialize all pointers to nullptr
        nodesEnd = new Node * [rows] {}; // Initialize all pointers to nullptr
        row_pointers = new int[rows + 1]();
        row_noe = new int[rows]();
    }

    ~CSR_Format() {


        delete[] row_pointers;
        delete[] row_noe;

        for (int i = 0; i < rows; ++i) {
            Node* current = nodesBegin[i];
            while (current) {
                Node* next = current->next;
                delete current;
                current = next;
            }
        }
        delete[] nodesBegin;
        delete[] nodesEnd;
    }

    void addElement(int row, int col, T value) {
        Node* newNode = new Node(col, value, nullptr);

        if (!nodesBegin[row]) {
            // If the first node in the row does not exist, add the new node
            nodesBegin[row] = newNode;
            nodesEnd[row] = newNode;  // Update the nodesEnd pointer
        }
        else {
            // If the first node already exists, find the right place to insert the new node
            Node* current = nodesBegin[row];
            Node* prev = nullptr;

            while (current && col > current->col) {
                prev = current;
                current = current->next;
            }

            if (prev && prev->col == col) {
                prev->value = value;
                return;
            }

            if (!prev) {

                // New node should be the first in the row
                newNode->next = current;
                nodesBegin[row] = newNode;  // Update the nodes pointer
            }
            else {

                // Insert the new node between the previous and current nodes
                prev->next = newNode;
                newNode->next = current;

                // If current is nullptr, the newNode is the new end of the row
                if (!current) {
                    nodesEnd[row] = newNode;  // Update the nodesEnd pointer
                }
            }
        }

        row_noe[row]++;  // Increase the number of elements in the row
        nnz++;
    }


    // Za brze dodavanje kod mnozenja Forward
    void addElementForward(int row, int col, T value) {
        Node* newNode = new Node(col, value, nullptr);

        if (!nodesBegin[row]) {
            // If the first node in the row does not exist, add the new node
            nodesBegin[row] = newNode;
            nodesEnd[row] = newNode;  // Update the nodesEnd pointer
        }
        else {
            // If the first node already exists, find the right place to insert the new node
            Node* current = nodesEnd[row];
            current->next = newNode;
            nodesEnd[row] = newNode;
        }

        row_noe[row]++;  // Increase the number of elements in the row
        nnz++;
    }


    const T& GetElement(int row, int col) const {
        if ((nodesBegin[row] && col < nodesBegin[row]->col) || (nodesEnd[row] && col > nodesEnd[row]->col))
            return zero;
        Node* current = nodesBegin[row];
        while (current) {
            if (current->col == col)
                return current->value;
            if (current->col > col)
                return zero;
            current = current->next;
        }
        return zero;
    }


    int getNNZ() const {
        return nnz;
    }

    const int* getRowPointers() const {
        for (int i = 1; i <= rows; ++i) {
            row_pointers[i] = row_pointers[i - 1] + row_noe[i - 1];
        }
        return row_pointers;
    }
    
    T* GetMatrix() {
        T* m = new T[rows * cols]();

        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            Node* current = nodesBegin[i];
            while (current) {
                m[i * rows + current->col] = current->value;
                current = current->next;
            }
        }

        return m;
    }


    std::pair<T*, int*> GetAllElementsWithColumns() {

        T* values = new T[nnz];
        int* indexColumns = new int[nnz];
        int index = 0;

        for (int i = 0; i < rows; i++) {
            Node* current = nodesBegin[i];
            //const int& rowBegin = row_pointers[i];
            while (current) {
                values[index] = current->value;
                indexColumns[index++] = current->col;
                current = current->next;
            }
        }
        return std::make_pair(values, indexColumns);
    }



    void printRowPointers() const {
        getRowPointers();
        std::cout << "Row Pointers: ";
        for (int i = 0; i <= rows; ++i) {
            std::cout << row_pointers[i] << " ";
        }
        std::cout << std::endl;
    }

    void printNodesBeginEnd() const {
        for (int i = 0; i < rows; i++) {
            auto node = nodesBegin[i];
            if (node) {
                auto end = nodesEnd[i];
                std::cout << "Pocetak reda " << i << " kolona " << node->col << " vrijednost " << node->value << " kraj reda kolona " << end->col << " vrijednost " << end->value << std::endl;
            }
        }
    }
};

template <typename T> class SparseMatrix {
private:
    int rows;
    int cols;
    CSR_Format<T>* csr;    
    CSC_Format<T>* csc;

public:
    SparseMatrix(int rows, int cols) : rows(rows), cols(cols) {
        csr = new CSR_Format<T>(rows, cols);
        csc = new CSC_Format<T>(rows, cols);
    }

    ~SparseMatrix() {
        delete csr;
        delete csc;
    }

    void addElement(int row, int col, T value) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            std::cout << "Ne može se dodati element sa tim indkesom u matricu." << std::endl;
            return;
        }
        csr->addElement(row, col, value);
        csc->addElement(row, col, value);
    }

    void addElementForward(int row, int col, T value, int m = 0) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            std::cout << "Ne može se dodati element sa tim indkesom u matricu." << std::endl;
            return;
        }
        csc->addElementForward(row, col, value);
        if(!m)
            csr->addElementForward(row, col, value);
    }

    int getNNZ() {
        return csr->getNNZ();
    }

    SparseMatrix<T>* multParallel(const SparseMatrix<T>& matrixB) const {
        if (cols != matrixB.rows) {
            std::cout << "Ne mogu se pomnoziti ove matrice." << std::endl;
            SparseMatrix<T>* r = new SparseMatrix<T>(0, 0);
            return r;
        }

        int row = rows;
        int col = matrixB.cols;

        const auto& rpA = csr->getRowPointers();
        const auto& A = csr->GetAllElementsWithColumns();
        const auto& valuesA = A.first;
        const auto& colsA = A.second;
        

        SparseMatrix<T>* result = new SparseMatrix<T>(row, col);
#pragma omp parallel for 
        for (int j = 0; j < col; ++j)
         {
            T* colB = matrixB.csc->GetColumn(j); 
            //auto startParallelForward = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
            for (int i = 0; i < rows; ++i) {

                T dotProduct = 0;
                // Racunaj skalarni produkt retka iz prve matrice i kolone iz druge matrice
                const int& nextRow = rpA[i + 1];
                
#pragma omp simd reduction(+:dotProduct)
                for (int k = rpA[i]; k < nextRow; ++k) {
                    const auto& valueB = colB[colsA[k]];
                    if (valueB != 0)
                        dotProduct += valuesA[k] * valueB;
                }

                if (dotProduct != 0) {

                    // Dodaj rezultat u rezultantnu matricu
                    result->addElementForward(i, j, dotProduct, 1);

                }

            }
            
            //auto endParallelForward = std::chrono::high_resolution_clock::now();
            //double parallelForwardTime = std::chrono::duration<double>(endParallelForward - startParallelForward).count();
            //std::cout << "Vrijeme za mnozenje jedne kolone sa svakim redom: " << parallelForwardTime << std::endl;
            delete[] colB;
        }

        return result;
    }



    

    SparseMatrix<T>* mult(const SparseMatrix<T>& matrixB) const {
        if (cols != matrixB.rows) {
            std::cout << "Ne mogu se pomnoziti ove matrice." << std::endl;
            SparseMatrix<T>* r = new SparseMatrix<T>(0, 0);
            return r;
        }

        int row = rows;
        int col = matrixB.cols;
        const auto& rpA = csr->getRowPointers();
        const auto& A = csr->GetAllElementsWithColumns();
        const auto& valuesA = A.first;
        const auto& colsA = A.second;
        

        SparseMatrix<T>* result = new SparseMatrix<T>(row, col);

        for (int j = 0; j < col; ++j)
        {
            // Izvuci cijelu kolonu iz matrice B
            T* colB = matrixB.csc->GetColumn(j);
            for (int i = 0; i < rows; ++i) {

                T dotProduct = 0;

                // Racunaj skalarni produkt retka iz prve matrice i kolone iz druge matrice
                const int& nextRow = rpA[i + 1];
                
                for (int k = rpA[i]; k < nextRow; ++k) {
                    const auto& valueB = colB[colsA[k]];
                    if (valueB != 0)
                        dotProduct += valuesA[k] * valueB;
                }

                if (dotProduct != 0)
                    // Dodaj rezultat u rezultantnu matricu
                    result->addElementForward(i, j, dotProduct, 1);
            }
            delete[] colB;
        }
        return result;
    }


    const T& GetElement(int row, int col) const {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw std::out_of_range("Pristup izvan opsega matrice");
        }
        return csc->GetElement(row, col);
    }


    // dvije različite funkcije za dobavljanje elemenata jedna linija u CSR_Format za 10^6 elemenata u dobije se ušteda od 13 s ako se treba cijela matrica ispisati da se zna gdje je nula a gdje nije.
    // print je bolji taj pogledati CSR_Format klasu
    void print() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                const auto a = csc->GetElement(i, j);
                std::cout << a << " ";
            }
            std::cout << std::endl;

        }
    }

    const int& getRows() {
        return rows;
    }

    const int& getCols() {
        return cols;
    }

};


// Dodavanje elemenata tako da prosječan broj elemenata svakog reda bude prosjecanBroj u matricu tipa SparseMatrix
template <typename T>
void dodajElementeURedove(SparseMatrix<T>& matrica, int n) {
    int rows = matrica.getRows();
    int cols = matrica.getCols();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < n; ++j) {
            matrica.addElementForward(i, j, 1);
        }
    }
}

template <typename T>
void dodajElementeURedove2(SparseMatrix<T>& matrica, int n) {
    int rows = matrica.getRows();
    int cols = matrica.getCols();
    for (int i = 0; i < matrica.getRows(); ++i) {
        for (int j = 0; j < n; ++j) {
            int randomCol = rand() % cols;
            matrica.addElement(i, randomCol, 1);
        }
    }
}


int main() {



    SparseMatrix<int> matrix2(3, 3);
    matrix2.addElementForward(0, 0, 1);
    matrix2.addElementForward(0, 1, 2);
    matrix2.addElementForward(0, 2, 3);
    matrix2.addElementForward(1, 0, 4);
    matrix2.addElementForward(1, 1, 5);
    matrix2.addElementForward(1, 2, 6);
    matrix2.addElementForward(2, 0, 7);
    matrix2.addElementForward(2, 1, 8);
    matrix2.addElementForward(2, 2, 9);

    auto rr = matrix2.multParallel(matrix2);

    rr->print();

    delete rr;

    int startSize = 1000;
    int endSize = 10000;
    std::vector<int> sizesVector;
    std::vector<double> sequentialForwardTimes;
    std::vector<double> parallelForwardTimes;
    std::vector<int> elementsRow;
    std::vector<int> nnz;
    for (int n = 10; n <= 1000; n *= 10) {
        for (int size = startSize; size <= endSize; size *= 10) {
            sizesVector.push_back(size);
            elementsRow.push_back(n);
            // Generirajte matrice za trenutnu veličinu
            SparseMatrix<int> m1(size, size);
            dodajElementeURedove(m1, n);
            nnz.push_back(m1.getNNZ());


            // Mjerenje vremena za sekvencijalno množenje unaprijed
            auto startSequentialForward = std::chrono::high_resolution_clock::now();
            //auto sequentialForwardResult = m1.mult(m1);
            auto endSequentialForward = std::chrono::high_resolution_clock::now();
            double sequentialForwardTime = std::chrono::duration<double>(endSequentialForward - startSequentialForward).count();
            //delete sequentialForwardResult;

            // Mjerenje vremena za paralelno množenje unaprijed
            auto startParallelForward = std::chrono::high_resolution_clock::now();
            auto parallelForwardResult = m1.multParallel(m1);
            auto endParallelForward = std::chrono::high_resolution_clock::now();
            double parallelForwardTime = std::chrono::duration<double>(endParallelForward - startParallelForward).count();
            delete parallelForwardResult;

            // Zabilježite rezultate u vektore
            sequentialForwardTimes.push_back(sequentialForwardTime);
            parallelForwardTimes.push_back(parallelForwardTime);
        }
        std::cout << "Gotovo mnozenje za n = " << n << std::endl;
    }

    // Ispis rezultata
    std::cout << "---------------------------------------------------------"<< std::endl;
    std::cout << " Rows | Seq. Forward | Par. Forward | El. in rows | nnz" << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    for (size_t i = 0; i < sizesVector.size(); ++i) {
        std::cout << sizesVector[i] << "   | "
            << sequentialForwardTimes[i] << " s       | " << parallelForwardTimes[i] << " s       | " << elementsRow[i] << " | " << nnz[i] << std::endl;
    }

    std::cout << "--------------------------------------------------------" << std::endl;

    return 0;

}