import requests
import time
import pandas as pd
from datetime import datetime


#AWS endpoint URL
API_ENDPOINT_URL = "http://fake-news-detector-env.eba-wdm367em.us-east-2.elasticbeanstalk.com/predict" 


TEST_CASES = {
    "fake_1": "Hi, I am a Nigerian Prince and in desperate need of your help!",
    "fake_2": "At least 3 dead, 11 injured in UPS plane crash on Pluto and governor says numbers likely to grow",
    "real_1": "At least 3 dead, 11 injured in UPS plane crash in Kentucky and governor says numbers likely to grow",
    "real_2": "Jays lose to Dodgers in game 7 of the world series"
}


def latency_test():
    print("Starting Latency Test...")
    
    
    results = []

    for case_name, text_input in TEST_CASES.items():
        print("Testing case: " + case_name)
        
        for i in range(100):
            payload = {"message": text_input}
            
            #get time before the call
            start_time = time.perf_counter()
            call_timestamp = datetime.now()
            try:
                #make post request
                response = requests.post(API_ENDPOINT_URL, json=payload, timeout=10)
                #calc latency
                latency = time.perf_counter() - start_time
                #record results
                results.append({
                    "test_case": case_name,
                    "iteration": i + 1,
                    "timestamp_utc": call_timestamp,
                    "latency_seconds": latency,
                    "prediction": response.json().get("label", "ERROR") if response.status_code == 200 else "ERROR"
                })
                
                #log progress every 20 calls
                if (i + 1) %20 == 0:
                    print("Completed call " + str(i+1) + " for " + case_name)

            except requests.exceptions.RequestException as e:
                print("Request failed for case " + case_name + " on iteration " + str(i+1) + ": " + str(e))

    print("Latency Test finished")
    
    #save results to csv file

    # convert list to dataframe
    df = pd.DataFrame(results)
    csv_filename = "api_latency_results.csv"
    df.to_csv(csv_filename, index=False)
    return df

if __name__ == "__main__":
    results_df = latency_test()
    
    if results_df is not None:
        print("\nResults Summary: ")
        print(f"Total calls: 400")
        print("\nAverage Latency: ")
        avg_latency = results_df.groupby("test_case")["latency_seconds"].mean()
        print(avg_latency)
        overall_avg = results_df["latency_seconds"].mean()
        print(f"\nOverall Average Latency: " + str(round(overall_avg, 4)) + " seconds")