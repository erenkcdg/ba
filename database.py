import psycopg2
from psycopg2 import OperationalError

def connect():
    return psycopg2.connect(
                host='localhost',
                port='5432',
                database='postgres',
                user='postgres', 
                password=''
            )
    
def save_settings(config, exp_config, beschreibung):
    try:
        connection = connect()
        if connection:
            ds                  = config['settings']['current_dataset']       
            num_epochs          = config['parameters_wbc']['num_epochs']
            batch_size          = config['parameters_wbc']['batch_size']
            learning_rate       = config['parameters_wbc']['learning_rate']
            general_seed        = config['parameters_wbc']['general_seed']
            num_trainings       = config['parameters_wbc']['num_trainings']
            length_dataset      = config['parameters_wbc']['length_dataset']
            current_model       = config['parameters_wbc']['current_model']
            if ds == 0:
                current_dataset = 'MNIST'
            else:
                current_dataset = 'BCW'
            noising             = config['opacus_wbc']['noising']
            clipping            = config['opacus_wbc']['clipping']
            make_private        = exp_config['settings']['make_private']
            data_processing     = exp_config['settings']['data_processing']
            dataset_mode        = exp_config['settings']['dataset_mode']
            init_mode           = exp_config['settings']['init_mode']
            dp_mode             = exp_config['settings']['dp_mode']
            cursor = connection.cursor()
            query = """
                INSERT INTO training_einstellung (num_epochs, batch_size, seed, noising, clipping, model, dataset, make_private, dataset_mode, data_processing, init_mode, dp_mode, num_trainings, length_dataset, learning_rate, beschreibung)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """
            cursor.execute(query, (num_epochs, batch_size, general_seed, noising, clipping, current_model, current_dataset, make_private, dataset_mode, data_processing, init_mode, dp_mode, num_trainings, length_dataset, learning_rate, beschreibung))  
            fetch_id = cursor.fetchone()
            if fetch_id is not None:
                new_id = fetch_id[0]
            else:
                new_id = None            
            connection.commit()
            return new_id

    except OperationalError as e:
        print(f"DB -->  save_einstellung(): Fehler beim Verbinden: {e}")
    except Exception as e:
        print(f"DB --> save_einstellung(): Fehler beim Einfügen des Datensatzes: {e}")
    finally:
        if 'connection' in locals() and connection:
            cursor.close()
            connection.close()

def save_training(run, training, loss, accuracy, epsilon, epsilon_batching, model_name):
    try:
        connection = connect()
        if connection:
            cursor = connection.cursor()
            query = """
                    INSERT INTO TRAINING(run, training, loss, accuracy, epsilon, epsilon_batching, model_name) 
                    VALUES(%s, %s, %s, %s, %s, %s, %s)
                    """
            cursor.execute(query, (run, training, loss, accuracy, epsilon, epsilon_batching, model_name))
            connection.commit()
    except OperationalError as e:
        print(f"DB --> save_training(): Fehler beim Verbinden: {e}")
    except Exception as e:
        print(f"DB --> save_training(): Fehler beim Einfügen des Datensatzes: {e}")
    finally:
        if 'connection' in locals() and connection:
            cursor.close()
            connection.close()

def save_distance(run,model1,model2,l2_distance):
    try:
        connection = connect()
        if connection:
            cursor = connection.cursor()
            l2_distance = float(l2_distance)
            query = """
                    INSERT INTO DISTANCE(run, model1, model2, distance) 
                    VALUES(%s, %s, %s, %s)
                    """
            cursor.execute(query, (run, model1, model2, l2_distance))
            connection.commit()
    except OperationalError as e:
        print(f"DB --> save_distance(): Fehler beim Verbinden: {e}")
    except Exception as e:
        print(f"DB --> save_distance(): Fehler beim Einfügen des Datensatzes: {e}")
    finally:
        if 'connection' in locals() and connection:
            cursor.close()
            connection.close()

def get_distances(run):
    try:
        connection = connect()
        if connection:
            cursor = connection.cursor()
            query = "SELECT distance FROM DISTANCE WHERE run = %s;"
            cursor.execute(query, (run,))
            distances = cursor.fetchall()
            return [distance[0] for distance in distances]

    except psycopg2.OperationalError as e:
        print(f"DB --> get_distances(): Fehler beim Verbinden: {e}")
    except Exception as e:
        print(f"DB --> get_distances(): Fehler beim Abrufen der Daten: {e}")
    finally:
        if 'connection' in locals() and connection:
            cursor.close()
            connection.close()

def get_one_distance(run, model):
    try:
        connection = connect()
        if connection:
            cursor = connection.cursor()
            query = "SELECT distance, model2 FROM DISTANCE WHERE run = %s and model1 = %s;"
            cursor.execute(query, (run,model,))
            results = cursor.fetchall()
            return [(distance[0], distance[1]) for distance in results]

    except psycopg2.OperationalError as e:
        print(f"DB --> get_distances(): Fehler beim Verbinden: {e}")
    except Exception as e:
        print(f"DB --> get_distances(): Fehler beim Abrufen der Daten: {e}")
    finally:
        if 'connection' in locals() and connection:
            cursor.close()
            connection.close()

def get_distances_with_model(run):
    try:
        connection = connect()
        if connection:
            cursor = connection.cursor()
            query = "select model1, model2, distance from distance where run = %s"
            cursor.execute(query, (run,))
            results = cursor.fetchall()
            return [(distance[0], distance[1], distance[2]) for distance in results]

    except psycopg2.OperationalError as e:
        print(f"DB --> get_distances(): Fehler beim Verbinden: {e}")
    except Exception as e:
        print(f"DB --> get_distances(): Fehler beim Abrufen der Daten: {e}")
    finally:
        if 'connection' in locals() and connection:
            cursor.close()
            connection.close()
