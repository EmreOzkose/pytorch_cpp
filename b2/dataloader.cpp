// https://github.com/pytorch/examples/blob/main/cpp/custom-dataset/custom-dataset.cpp

#include <torch/torch.h>


class HouseDataset : public torch::data::datasets::Dataset<HouseDataset> {
    using Example = torch::data::Example<>;
    std::vector<std::vector<std::string>> content_house_prices = read_dataset_csv();

    public:
    HouseDataset() {}

    Example get(size_t index) {
        /*
        for(int i=0;i<content_house_prices.size();i++){
            for(int j=0;j<content_house_prices[i].size();j++){
                std::cout<<content_house_prices[i][j]<<" ";
            }
            std::cout<<"\n";
        }
        */
        
        float lot_area = stoi(content_house_prices[index][0]);
        float built_year = stoi(content_house_prices[index][1]);
        float target = stoi(content_house_prices[index][2]);
        return {torch::tensor({lot_area, built_year}), torch::tensor({target})};
    }

    torch::optional<size_t> size() const {
        return content_house_prices.size();
    }

    std::vector<std::vector<std::string>> read_dataset_csv(){
        // https://java2blog.com/read-csv-file-in-cpp
        std::vector<std::vector<std::string>> content;
        std::vector<std::string> row;
        std::string line, word;
        
        std::string fname = "/path/to/minimal_example.csv";
        std::fstream file (fname, std::ios::in);
        if(file.is_open()){
            while(getline(file, line))
            {
            row.clear();
            
            std::stringstream str(line);
            
            while(getline(str, word, ','))
                row.push_back(word);
                content.push_back(row);
            }
            std::cout<<"Reading dataset is done! Size is " << content_house_prices.size() << std::endl;
        }
        else
            std::cout<<"Could not open the file" << std::endl;

        return content;
    }
};
