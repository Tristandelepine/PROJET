import sys
import os
from gurobipy import Model, GRB, GurobiError

dataset_path = "C:/Users/trist/OneDrive/Documents/Cours/Gurobi/videos/datasets/trending_4000_10k.in"

class Video:
    """Représente une vidéo (taille $s_v$)."""
    def __init__(self, id, size):
        self.id = id
        self.size = size 
        
class Endpoint:
    """Représente un point d'accès client (latence $l_{e}^{D}$)."""
    def __init__(self, id, dc_latency, cache_connections):
        self.id = id
        self.dc_latency = dc_latency 
        self.cache_connections = cache_connections 

class Request:
    """Représente un ensemble de requêtes ($n_r$) pour une vidéo ($v_r$) et un endpoint ($e_r$)."""
    def __init__(self, id, video_id, endpoint_id, count):
        self.id = id
        self.video_id = video_id
        self.endpoint_id = endpoint_id
        self.count = count

class ProblemData:
    """Conteneur pour toutes les données lues."""
    def __init__(self):
        self.videos = {}       
        self.endpoints = {}    
        self.caches = {}       
        self.requests = {}     
        self.cache_capacity = 0

def read_data(filepath):
    """Lit les données du fichier d'entrée (format Hash Code 2017) avec une limite de requêtes."""
    print(f"--- Lecture du fichier de données : {filepath} ---")
    data = ProblemData()

    MAX_REQUESTS_TO_PROCESS = 5000 
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            V, E, R_total, C, X = map(int, f.readline().split())
            
            R_processed = min(R_total, MAX_REQUESTS_TO_PROCESS)
            
            data.cache_capacity = X
            data.caches = {c_id: X for c_id in range(C)}
            print(f"  > V: {V}, E: {E}, R: {R_total} (Limite fixée à {R_processed}), C: {C}, Capacité Cache (X): {X}")

            sizes = list(map(int, f.readline().split()))
            for v_id, size in enumerate(sizes):
                data.videos[v_id] = Video(v_id, size)
            print("  > Tailles des vidéos lues.")

            for e_id in range(E):
                line = f.readline().split()
                if not line: raise EOFError("Fin de fichier inattendue.")
                dc_latency, nb_cache_conn = map(int, line)

                cache_connections = {}
                for _ in range(nb_cache_conn):
                    c_id, c_latency = map(int, f.readline().split())
                    cache_connections[c_id] = c_latency
                
                data.endpoints[e_id] = Endpoint(e_id, dc_latency, cache_connections)
            print(f"  > {E} Endpoints lus.")

            for r_id in range(R_total):
                v_id, e_id, count = map(int, f.readline().split())
                
                if r_id < R_processed:
                    data.requests[r_id] = Request(r_id, v_id, e_id, count)
                    
            print(f"  > {len(data.requests)} Requêtes lues et traitées.")

    except Exception as e:
        print(f"Erreur critique lors de la lecture des données : {e}")
        sys.exit(1)
    
    print("--- Lecture des données terminée. ---\n")
    return data

def write_solution(data, model, x_vars, output_filepath="videos.out"):
    """Écrit la configuration optimale des caches dans le fichier de sortie, en utilisant le modèle."""
    print(f"--- Écriture de la solution dans {output_filepath} ---")
    
    cache_to_videos = {c_id: [] for c_id in data.caches}
    
    try:
        x_solution_values = model.getAttr(GRB.Attr.X, list(x_vars.values()))
        
        for var_name, var_value in zip(x_vars.keys(), x_solution_values):
            
            if var_value > 0.5:
                parts = var_name.split('_')
                v_id = int(parts[1])
                c_id = int(parts[2])
                
                cache_to_videos[c_id].append(v_id)

    except GurobiError as e:
        print(f"Erreur Gurobi lors de la lecture de la solution: {e.message}")
        return
    except Exception as e:
        print(f"Erreur inattendue lors de l'écriture de la solution: {e}")
        return

    filled_caches = {c_id: v_list for c_id, v_list in cache_to_videos.items() if v_list}


    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(f"{len(filled_caches)}\n") 
        
        for c_id, v_list in filled_caches.items():
            v_list.sort() 
            line = f"{c_id} " + " ".join(map(str, v_list))
            f.write(f"{line}\n")

    print("--- Écriture de la solution terminée. ---")

def solve_mip(data):
    """Crée et résout le modèle MIP de maximisation du gain de latence."""
    print("--- Construction du modèle MIP Gurobi ---")
    
    try:
        model = Model("Video_Caching_MIP")
        
        x_vars = {}
        for v_id in data.videos:
            for c_id in data.caches:
                x_vars[f'x_{v_id}_{c_id}'] = model.addVar(vtype=GRB.BINARY, name=f'x_{v_id}_{c_id}')

        y_vars = {}
        for r_id, req in data.requests.items():
            e_id = req.endpoint_id
            for c_id in data.endpoints[e_id].cache_connections:
                y_vars[f'y_{r_id}_{c_id}'] = model.addVar(vtype=GRB.BINARY, name=f'y_{r_id}_{c_id}')
        
        model.update()
        print(f"  > {len(x_vars)} variables de stockage (x) et {len(y_vars)} variables de service (y) créées.")

        objective = 0
        for r_id, req in data.requests.items():
            e_id = req.endpoint_id
            endpoint = data.endpoints[e_id]
            
            dc_latency = endpoint.dc_latency
            
            for c_id, c_latency in endpoint.cache_connections.items():
                latency_saved = dc_latency - c_latency
                y_var = y_vars[f'y_{r_id}_{c_id}']
                objective += req.count * latency_saved * y_var

        model.setObjective(objective, GRB.MAXIMIZE)
        print("  > Objectif (Maximiser le gain de latence) défini.")
        
        for c_id in data.caches:
            cache_capacity = data.cache_capacity
            lhs = 0
            for v_id in data.videos:
                lhs += data.videos[v_id].size * x_vars[f'x_{v_id}_{c_id}']
            model.addConstr(lhs <= cache_capacity, name=f'CacheCap_{c_id}')
        print("  > Contraintes de capacité des caches ajoutées.")

        for r_id, req in data.requests.items():
            v_id = req.video_id
            e_id = req.endpoint_id
            for c_id in data.endpoints[e_id].cache_connections:
                y_var = y_vars[f'y_{r_id}_{c_id}']
                x_var = x_vars[f'x_{v_id}_{c_id}']
                model.addConstr(y_var <= x_var, name=f'Link_{r_id}_{c_id}')
        print("  > Contraintes de liaison y_rc <= x_vc ajoutées.")

        for r_id, req in data.requests.items():
            e_id = req.endpoint_id
            lhs = 0
            for c_id in data.endpoints[e_id].cache_connections:
                lhs += y_vars[f'y_{r_id}_{c_id}']
            model.addConstr(lhs <= 1, name=f'UniqueService_{r_id}')
        print("  > Contraintes de service unique ajoutées (<= 1).")

        print("\n--- Résolution du modèle ---")
        model.Params.TimeLimit = 3600 
        model.Params.MIPGap = 5e-3
        model.Params.LogFile = "gurobi_log.txt"
        
        model.write("videos.mps")
        print("  > Modèle écrit dans videos.mps.")

        model.optimize()
        
        if model.SolCount == 0:
            if model.Status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
                 print("\n  > Le modèle est infaisable ou non borné. Aucune solution trouvée.")
            else:
                 print(f"\n  > Le solveur a terminé avec le statut: {model.Status}. Aucune solution entière réalisable trouvée.")
            return None
        
        elif model.Status == GRB.OPTIMAL:
            print(f"\n  > Solution OPTIMALE trouvée. Gap: {model.MIPGap * 100:.3f}%")
            print(f"  > Valeur optimale de l'objectif (Gain de Latence): {model.ObjVal:.2f}")
            return model, x_vars 
            
        elif model.MIPGap <= 5e-3:
            print(f"\n  > Solution trouvée (Statut: {model.Status}). Gap ({model.MIPGap * 100:.3f}%) respecté.")
            print(f"  > Valeur de l'objectif (Gain de Latence): {model.ObjVal:.2f}")
            return model, x_vars
        
        else:
            print(f"\n  > Solution trouvée (Statut: {model.Status}), mais le Gap ({model.MIPGap * 100:.3f}%) est supérieur à 0.5% (5e-3).")
            return None

    except GurobiError as e:
        print(f"Erreur Gurobi: {e.message}")
        print(f"\n*** ÉCHEC DE LA LICENCE/TAILLE DU MODÈLE. Veuillez vérifier votre licence Gurobi. ***")
        return None
    except Exception as e:
        print(f"Une erreur inattendue s'est produite: {e}")
        return None

if __name__ == "__main__":
    
    print("--- Lancement du script d'optimisation (Mode Interactif) ---")
    
    if not dataset_path:
        print("Erreur : Aucun chemin de dataset fourni. Arrêt du script.")
        sys.exit(1)
        
    problem_data = read_data(dataset_path)
    
    result = solve_mip(problem_data)
    
    if result:
        optimal_model, optimal_x_vars = result
        write_solution(problem_data, optimal_model, optimal_x_vars, output_filepath="videos.out")
    else:
        print("\n*** Échec de la résolution ou Gap non atteint. Le fichier videos.out n'a pas été créé. ***")
